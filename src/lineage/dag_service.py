from typing import List, Optional

import networkx as nx
import requests
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Healthcare Data Lineage DAG Service")

MARQUEZ_URL = "http://localhost:4601/api/v1"
G = nx.DiGraph()


def get_all_namespaces() -> List[str]:
    """Get all namespaces from Marquez"""
    try:
        response = requests.get(f"{MARQUEZ_URL}/namespaces")
        if response.status_code == 200:
            return [ns["name"] for ns in response.json().get("namespaces", [])]
    except Exception as e:
        print(f"Error fetching namespaces: {e}")
    return []


def parse_layer_from_path(path: str) -> str:
    """Extract layer from file path or dataset name"""
    path_lower = path.lower()

    if "bronze" in path_lower:
        return "bronze"
    elif "silver" in path_lower:
        return "silver"
    elif "gold" in path_lower:
        return "gold"
    elif "metric" in path_lower or "final" in path_lower:
        return "metric"

    return "unknown"


def build_global_graph() -> nx.DiGraph:
    """Build graph from Marquez lineage data"""
    global G
    G = nx.DiGraph()

    print("\n" + "=" * 60)
    print("BUILDING LINEAGE GRAPH FROM MARQUEZ")
    print("=" * 60)

    all_namespaces = get_all_namespaces()
    print(f"\nFound {len(all_namespaces)} namespaces")

    dataset_nodes = set()
    edges_created = 0

    # Fetch all jobs from all namespaces
    for namespace in all_namespaces:
        try:
            jobs_response = requests.get(f"{MARQUEZ_URL}/namespaces/{namespace}/jobs")
            if jobs_response.status_code != 200:
                continue

            jobs = jobs_response.json().get("jobs", [])

            for job in jobs:
                job_name = job["name"]

                # Get detailed job info
                job_url = f"{MARQUEZ_URL}/namespaces/{namespace}/jobs/{job_name}"
                job_response = requests.get(job_url)

                if job_response.status_code != 200:
                    continue

                job_data = job_response.json()

                # Get inputs and outputs
                inputs = job_data.get("inputs", [])
                outputs = job_data.get("outputs", [])

                # If this job has lineage, process it
                if inputs or outputs:
                    # Add input datasets as nodes
                    for inp in inputs:
                        inp_namespace = inp.get("namespace", "unknown")
                        inp_name = inp.get("name", "unknown")
                        inp_id = f"{inp_namespace}/{inp_name}"

                        if inp_id not in dataset_nodes:
                            G.add_node(
                                inp_id,
                                id=inp_id,
                                name=inp_name,
                                namespace=inp_namespace,
                                layer=parse_layer_from_path(inp_name),
                                type="dataset",
                            )
                            dataset_nodes.add(inp_id)

                    # Add output datasets as nodes
                    for out in outputs:
                        out_namespace = out.get("namespace", "unknown")
                        out_name = out.get("name", "unknown")
                        out_id = f"{out_namespace}/{out_name}"

                        if out_id not in dataset_nodes:
                            G.add_node(
                                out_id,
                                id=out_id,
                                name=out_name,
                                namespace=out_namespace,
                                layer=parse_layer_from_path(out_name),
                                type="dataset",
                            )
                            dataset_nodes.add(out_id)

                    # Create edges: input -> output (data flow)
                    for inp in inputs:
                        inp_id = f"{inp.get('namespace', 'unknown')}/{inp.get('name', 'unknown')}"

                        for out in outputs:
                            out_id = f"{out.get('namespace', 'unknown')}/{out.get('name', 'unknown')}"

                            if not G.has_edge(inp_id, out_id):
                                G.add_edge(
                                    inp_id,
                                    out_id,
                                    etype="transform",
                                    job=f"{namespace}/{job_name}",
                                )
                                edges_created += 1

        except Exception as e:
            print(f"Error processing namespace {namespace}: {e}")

    # Stats
    layer_counts = {}
    for node, data in G.nodes(data=True):
        layer = data.get("layer", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print("\n✓ Graph built:")
    print(f"  Datasets: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  By layer: {layer_counts}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print("=" * 60)

    return G


@app.on_event("startup")
async def startup_event():
    """Load graph on startup"""
    print("Loading lineage from Marquez...")
    build_global_graph()
    print("\n✓ DAG Service ready!")


@app.get("/")
async def root():
    """Health check"""
    layers = {}
    for node, data in G.nodes(data=True):
        layer = data.get("layer", "unknown")
        layers[layer] = layers.get(layer, 0) + 1

    return {
        "service": "Healthcare Data Lineage DAG Service",
        "status": "running",
        "graph_stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "datasets_by_layer": layers,
        },
    }


@app.get("/graph/refresh")
async def refresh_graph():
    """Reload graph from Marquez"""
    build_global_graph()
    return {
        "status": "refreshed",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


@app.get("/graph/stats")
async def get_graph_stats():
    """Get detailed graph statistics"""
    layers = {}
    namespaces = {}

    for node, data in G.nodes(data=True):
        layer = data.get("layer", "unknown")
        layers[layer] = layers.get(layer, 0) + 1

        ns = data.get("namespace", "unknown")
        namespaces[ns] = namespaces.get(ns, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "by_layer": layers,
        "by_namespace": namespaces,
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


@app.get("/graph/docs")
async def get_rag_documents(
    namespace: Optional[str] = None,
    layer: Optional[str] = Query(
        None, description="Filter by layer: bronze, silver, gold, metric"
    ),
):
    """Get text documents for RAG retrieval"""
    documents = []

    for node, data in G.nodes(data=True):
        if namespace and data.get("namespace") != namespace:
            continue

        if layer and data.get("layer") != layer:
            continue

        texts = []
        texts.append(f"{data.get('name', node)}")
        texts.append(f"Layer: {data.get('layer', 'unknown')}")
        texts.append(f"Namespace: {data.get('namespace', 'unknown')}")

        incoming = list(G.predecessors(node))
        outgoing = list(G.successors(node))

        texts.append(f"Incoming: {len(incoming)}, Outgoing: {len(outgoing)}")

        if incoming:
            incoming_names = [
                G.nodes[n].get("name", n).split("/")[-1] for n in incoming[:5]
            ]
            texts.append(f"Consumes: {', '.join(incoming_names)}")

        if outgoing:
            outgoing_names = [
                G.nodes[n].get("name", n).split("/")[-1] for n in outgoing[:5]
            ]
            texts.append(f"Feeds into: {', '.join(outgoing_names)}")

        documents.append({"id": node, "texts": texts})

    return documents


@app.post("/graph/subgraph")
async def extract_subgraph(node_ids: List[str]):
    """Extract subgraph for given node IDs"""
    if not node_ids:
        raise HTTPException(status_code=400, detail="node_ids cannot be empty")

    subgraph_nodes = set(node_ids)

    for node_id in node_ids:
        if node_id in G:
            subgraph_nodes.update(G.predecessors(node_id))
            subgraph_nodes.update(G.successors(node_id))

    subgraph = G.subgraph(subgraph_nodes).copy()

    nodes_data = [{"id": n, **G.nodes[n]} for n in subgraph.nodes()]
    edges_data = [
        {"source": s, "target": t, **d} for s, t, d in subgraph.edges(data=True)
    ]

    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "stats": {
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
        },
    }


@app.get("/graph/node/{node_id:path}")
async def get_node_details(node_id: str):
    """Get detailed node information"""
    if node_id not in G:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    node_data = G.nodes[node_id]
    incoming = list(G.predecessors(node_id))
    outgoing = list(G.successors(node_id))

    return {
        "id": node_id,
        "attributes": dict(node_data),
        "incoming": [
            {
                "id": n,
                "name": G.nodes[n].get("name", n),
                "layer": G.nodes[n].get("layer", "unknown"),
            }
            for n in incoming
        ],
        "outgoing": [
            {
                "id": n,
                "name": G.nodes[n].get("name", n),
                "layer": G.nodes[n].get("layer", "unknown"),
            }
            for n in outgoing
        ],
    }


@app.get("/graph/cycles")
async def find_cycles():
    """Find cycles in the graph"""
    if nx.is_directed_acyclic_graph(G):
        return {"has_cycles": False, "message": "Graph is a valid DAG"}

    try:
        cycles = list(nx.simple_cycles(G))

        cycle_info = []
        for cycle in cycles[:10]:  # Show first 10 cycles
            cycle_nodes = []
            for node in cycle:
                cycle_nodes.append(
                    {
                        "id": node,
                        "name": G.nodes[node].get("name", node),
                        "layer": G.nodes[node].get("layer", "unknown"),
                    }
                )
            cycle_info.append(cycle_nodes)

        return {
            "has_cycles": True,
            "total_cycles": len(cycles),
            "sample_cycles": cycle_info,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/graph/fix_cycles")
async def fix_cycles():
    """Remove edges to break cycles and make it a DAG"""
    global G

    if nx.is_directed_acyclic_graph(G):
        return {"message": "Already a DAG", "edges_removed": 0}

    edges_removed = 0
    removed_edges = []

    try:
        while not nx.is_directed_acyclic_graph(G):
            cycle = nx.find_cycle(G)

            if cycle:
                src, dst = cycle[0][0], cycle[0][1]
                G.remove_edge(src, dst)
                edges_removed += 1
                removed_edges.append(f"{src} -> {dst}")

        return {
            "message": "Cycles fixed",
            "edges_removed": edges_removed,
            "is_dag": nx.is_directed_acyclic_graph(G),
            "removed_edges": removed_edges[:20],  # Show first 20
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/graph/export")
async def export_graph():
    """Export entire graph for visualization or storage"""
    nodes = []
    for node, data in G.nodes(data=True):
        nodes.append(
            {
                "id": node,
                "name": data.get("name", node),
                "short_name": data.get("name", node).split("/")[-1],
                "layer": data.get("layer", "unknown"),
                "namespace": data.get("namespace", "unknown"),
            }
        )

    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append(
            {"source": src, "target": dst, "type": data.get("etype", "transform")}
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "is_dag": nx.is_directed_acyclic_graph(G),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
