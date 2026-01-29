# %%
import json

from index import RAGIndex
from prompt import llm
from subgraph import (
    build_graph,
    draw_full_with_subgraph_overlay,
    get_min_connecting_subgraph,
    subgraph_texted,
)

JSON_PATHS = {
    1: "/srv/bigdata/data/custom/1_cms_unified_lineage_corrected.json",
    2: "/srv/bigdata/data/custom/2_cost_analytics_lineage.json",
    3: "/srv/bigdata/data/custom/3_geographic_health_equity_lineage.json",
    4: "/srv/bigdata/data/custom/4_provider_performance_network_lineage.json",
    5: "/srv/bigdata/data/custom/5_payer_utilization_intelligence_lineage.json",
}

TOPK = 3


def main(prompt: str = None, data_id: int = 1, topk: int = TOPK):
    """
    Args:
        prompt: Question (e.g., "What is the cost analysis?")
        data_id: Data ID (1-5, default: 1)
        topk: Number of top-k relevant edges to retrieve (default: 3)
    """
    # Check prompt
    if prompt is None or not prompt.strip():
        print("[ERROR] Please provide a question.")
        print("Usage: main(prompt='question content', data_id=1, topk=3)")
        return

    prompt = prompt.strip()

    # Check data ID
    if data_id not in JSON_PATHS:
        print(f"[ERROR] Invalid data ID: {data_id}. Please enter a number between 1-5.")
        print("\nAvailable data IDs:")
        for did, path in JSON_PATHS.items():
            filename = path.split("/")[-1]
            print(f"  {did}: {filename}")
        return

    path = JSON_PATHS.get(data_id)

    # Print selected file information
    selected_filename = path.split("/")[-1]
    print(f"Selected data file: {selected_filename}")
    print(f"File path: {path}\n")

    print("\n" + "=" * 70)
    print("Answer for Original Prompt")
    print("=" * 70)
    print(f"Question: {prompt}\n")

    # 1) Answer for original prompt
    answer_original = llm(prompt=prompt)
    print(f"Answer:\n{answer_original}\n")

    print("\n" + "=" * 70)
    print("Answer for Prompt with Subgraph")
    print("=" * 70)

    # 2) Load graph and build RAG index
    print("Loading graph...")
    with open(path, "r") as f:
        data = json.load(f)

    graph = build_graph(data)
    edges = []
    for u, v, attr in graph.edges(data=True):
        edges.append({"id": f"{u}:{v}", "text": attr.get("operations")})

    print("Building RAG index...")
    rag_index = RAGIndex()
    rag_index.build(edges)

    # 3) Search and extract subgraph
    try:
        results = rag_index.search(prompt, topk=topk)
        print(f"Top-{len(results)} relevant edges found (topk={topk})")

        if not results:
            print("[WARNING] No relevant edges found. Using original prompt only.")
            answer_with_subgraph = answer_original
        else:
            subgraph = get_min_connecting_subgraph(results, graph, directed=True)
            print(f"Subgraph: contains {subgraph.number_of_edges()} edges")

            # Visualize subgraph
            if subgraph.number_of_nodes() > 0 or subgraph.number_of_edges() > 0:
                print("\nDrawing subgraph visualization...")
                draw_full_with_subgraph_overlay(graph, subgraph)

            # 4) Generate prompt with subgraph
            if subgraph.number_of_edges() == 0:
                print("[WARNING] Subgraph is empty. Using original prompt only.")
                answer_with_subgraph = answer_original
            else:
                texts_list = subgraph_texted(subgraph)
                texts_list.append(f"The Question is: {prompt}")
                texts = str(texts_list)

                print(
                    "\nGenerating answer with prompt including subgraph information...\n"
                )
                answer_with_subgraph = llm(prompt=texts)
    except Exception as e:
        print(f"[ERROR] Error during subgraph extraction: {e}")
        print("Falling back to original answer.")
        answer_with_subgraph = answer_original

    print(f"Answer:\n{answer_with_subgraph}\n")
    print("=" * 70)


if __name__ == "__main__":
    # Enter values directly here to use
    # Examples:
    # main(prompt="What is the cost analysis?", data_id=1, topk=3)
    # main(prompt="How does geographic health equity work?", data_id=3, topk=5)

    main(
        prompt="What are the risk_tier classifications in gold_predictive_risk?",
        data_id=3,
        topk=3,
    )

# %%
