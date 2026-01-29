import json

import requests

BASE_URL = "http://localhost:8000"

# 1. Check health
print("1. Health Check")
response = requests.get(f"{BASE_URL}/")
print(json.dumps(response.json(), indent=2))

# 2. Get graph stats
print("\n2. Graph Statistics")
response = requests.get(f"{BASE_URL}/graph/stats")
print(json.dumps(response.json(), indent=2))

# 3. Get RAG documents
print("\n3. RAG Documents (Bronze layer)")
response = requests.get(f"{BASE_URL}/graph/docs?layer=bronze")
docs = response.json()
print(f"Found {len(docs)} bronze layer documents")
print("Sample:", json.dumps(docs[0], indent=2))

# 4. Get subgraph
print("\n4. Extract Subgraph")
node_ids = [docs[0]["id"], docs[1]["id"]] if len(docs) >= 2 else [docs[0]["id"]]
response = requests.post(f"{BASE_URL}/graph/subgraph", json=node_ids)
subgraph = response.json()
print(
    f"Subgraph: {subgraph['stats']['node_count']} nodes, {subgraph['stats']['edge_count']} edges"
)

# 5. Get specific namespace datasets
print("\n5. Geographic Health Equity Datasets")
response = requests.get(f"{BASE_URL}/namespaces/geographic_health_equity/datasets")
print(json.dumps(response.json(), indent=2))
