# %%
import json
from itertools import combinations

import faiss
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from groq import Groq

# import faiss
from sentence_transformers import SentenceTransformer


class RAGIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []
        self.emb: np.ndarray | None = None

    def build(self, docs):
        self.docs = docs
        texts = [d["text"] for d in docs]
        emb = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        self.emb = emb
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self.index = index

    def search(self, query: str, topk: int = 8):
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, topk)
        return [self.docs[i] for i in I[0]]


def build_graph(data):
    G = nx.DiGraph()

    for t in data["transformations"]:
        child = t["output"]
        parents = t.get("inputs", [])
        ops = t.get("operations", [])
        desc = t.get("description", "")

        # operations → str(list or dict)
        ops_str = str(ops)  # ★★★ Key: stored as "[{'op':'join'}, {'op':'agg'}]" format

        for p in parents:
            G.add_edge(
                p,
                child,
                operations=ops_str,  # Python repr string
                description=desc,
                op_count=len(ops),
            )

    return G


def get_min_connecting_subgraph(
    related_edges, full_graph: nx.DiGraph, directed: bool = True
):
    """
    related_edges: [{'id': 'u:v', 'text': '...'}, ...]
    full_graph   : Full DAG (nx.DiGraph)
    directed     : If True, consider direction in shortest_path, if False, use undirected shortest_path
    """
    # 1️⃣ Collect both end nodes of selected edges into terminal set
    terminals = set()
    base_edges = []

    for edge in related_edges:
        u, v = edge["id"].split(":", 1)
        if full_graph.has_edge(u, v):
            terminals.add(u)
            terminals.add(v)
            base_edges.append((u, v))

    if not terminals:
        return nx.DiGraph()

    # 2️⃣ Select graph based on directionality
    Gpath = full_graph if directed else full_graph.to_undirected()

    # 3️⃣ For minimum connectivity:
    #    - Find shortest path between each terminal pair
    #    - Collect all nodes/edges included in those paths into a union
    nodes_in_sub = set()
    edges_in_sub = set()

    # First, always include the selected edges themselves
    for u, v in base_edges:
        nodes_in_sub.add(u)
        nodes_in_sub.add(v)
        edges_in_sub.add((u, v))

    # Try shortest path for each terminal pair
    for s, t in combinations(terminals, 2):
        if s not in Gpath or t not in Gpath:
            continue
        try:
            path = nx.shortest_path(Gpath, source=s, target=t)
        except nx.NetworkXNoPath:
            continue

        # Add all nodes/edges on the path
        nodes_in_sub.update(path)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            # Find edge matching direction based on full_graph
            if full_graph.has_edge(a, b):
                edges_in_sub.add((a, b))
            elif full_graph.has_edge(b, a):
                edges_in_sub.add((b, a))
            # Skip if neither exists (theoretically rare)

    # 4️⃣ Create actual subgraph (copy attributes from full_graph)
    subgraph = nx.DiGraph()
    for n in nodes_in_sub:
        if n in full_graph:
            subgraph.add_node(n, **full_graph.nodes[n])

    for u, v in edges_in_sub:
        if full_graph.has_edge(u, v):
            subgraph.add_edge(u, v, **full_graph.edges[u, v])

    return subgraph


def draw_full_with_subgraph_overlay(full_graph, sub_graph, figsize=(14, 12)):
    plt.figure(figsize=figsize)

    # 1️⃣ Calculate full graph layout once
    pos = nx.spring_layout(full_graph, k=0.7, seed=42)

    # 2️⃣ Full graph (default color: light gray)
    nx.draw(
        full_graph,
        pos,
        node_size=600,
        node_color="#DDDDDD",
        edge_color="#CCCCCC",
        with_labels=True,
        font_size=8,
        arrows=True,
    )

    # 3️⃣ Highlighted subgraph (nodes/edges in different colors)
    nx.draw(
        full_graph,  # draw from full graph but
        pos,
        nodelist=list(sub_graph.nodes()),
        edgelist=list(sub_graph.edges()),
        node_size=700,
        node_color="#FFCC00",  # 🔥 Yellow highlight
        edge_color="#FF8800",  # 🔥 Orange edges
        with_labels=True,
        font_size=9,
        arrows=True,
        width=2.5,  # Thick
    )

    plt.title("Full Graph + Subgraph Highlight Overlay")
    plt.axis("off")
    plt.show()


def llm(prompt):
    """Generate response using Groq API"""
    try:
        client = Groq(api_key="Groq API Key")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def subgraph_texted(graph):
    texts = []
    for u, v, attr in graph.edges(data=True):
        ops = attr.get("operations")  # Operations string/list we stored
        texts.append(f"input node {u} became {v} through operations: {ops}")
    return texts


def run_single_row(path: str, prompt: str, annotation: str, topk: int = 3):
    """
    For a single row:
      - First Groq answer (no subgraph)
      - Answer with subgraph RAG
      - Cosine similarity between annotation and each answer
    Calculate and return these.
    """
    # print("=" * 50)
    # print("Groq AI Chat without Subgraph Extraction")
    # print("Prompt:", prompt)

    # 1) First answer (no subgraph)
    answer_fir = llm(prompt=prompt)
    # print("Groq:", answer_fir)
    # print("-" * 50)

    # 2) Load graph for RAG and build index
    with open(path, "r") as f:
        data = json.load(f)

    graph = build_graph(data)
    edges = []
    for u, v, attr in graph.edges(data=True):
        edges.append({"id": f"{u}:{v}", "text": attr.get("operations")})

    # print("Building RAG Index...")
    RagIndex = RAGIndex()
    RagIndex.build(edges)

    # 3) Search & extract subgraph
    results = RagIndex.search(prompt, topk=topk)
    # print(f"Top-{len(results)} relevant edges:")
    subgraph = get_min_connecting_subgraph(results, graph, directed=True)
    # print(f"Subgraph has {subgraph.number_of_edges()} edges.")
    # print("Subgraph edges with descriptions:")

    texts_list = subgraph_texted(subgraph)
    texts_list.append(f"The Question is:{prompt}")
    # Text to put in RAG prompt
    texts = str(texts_list)

    # print("\nGroq: ", end="")
    answer = llm(prompt=texts)

    # print("Groq AI with Subgraph Extraction")
    # print("Groq:", answer)
    # print("=" * 50)

    # 4) Calculate cosine similarity
    anno_vec = RagIndex.model.encode(annotation, normalize_embeddings=True)
    ansf_vec = RagIndex.model.encode(answer_fir, normalize_embeddings=True)
    ans_vec = RagIndex.model.encode(answer, normalize_embeddings=True)

    sim_first = float(np.dot(anno_vec, ansf_vec))
    sim_subgraph = float(np.dot(anno_vec, ans_vec))

    # print("Cosine Similarity (Annotation vs. First Answer):", sim_first)
    # print("Cosine Similarity (Annotation vs. Subgraph Answer):", sim_subgraph)

    return sim_first, sim_subgraph
