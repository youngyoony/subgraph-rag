import json
import os
from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "./final_dag_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EXPORTING FINAL DAG OUTPUTS")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Export full graph
print("\n1. Exporting full graph...")
response = requests.get(f"{BASE_URL}/graph/export")
with open(f"{OUTPUT_DIR}/full_graph_{timestamp}.json", "w") as f:
    json.dump(response.json(), f, indent=2)
print(f"✓ Saved full_graph_{timestamp}.json")

# 2. Export RAG documents by layer
print("\n2. Exporting RAG documents by layer...")
for layer in ["bronze", "silver", "gold", "unknown"]:
    response = requests.get(f"{BASE_URL}/graph/docs?layer={layer}")
    docs = response.json()

    with open(f"{OUTPUT_DIR}/rag_docs_{layer}_{timestamp}.json", "w") as f:
        json.dump(docs, f, indent=2)

    print(f"✓ Saved rag_docs_{layer}_{timestamp}.json ({len(docs)} documents)")

# 3. Export all RAG documents (combined)
print("\n3. Exporting all RAG documents...")
response = requests.get(f"{BASE_URL}/graph/docs")
all_docs = response.json()

with open(f"{OUTPUT_DIR}/rag_docs_all_{timestamp}.json", "w") as f:
    json.dump(all_docs, f, indent=2)
print(f"✓ Saved rag_docs_all_{timestamp}.json ({len(all_docs)} documents)")

# 4. Export graph statistics
print("\n4. Exporting statistics...")
response = requests.get(f"{BASE_URL}/graph/stats")
stats = response.json()

with open(f"{OUTPUT_DIR}/graph_stats_{timestamp}.json", "w") as f:
    json.dump(stats, f, indent=2)
print("✓ Saved graph_stats.json")

# 5. Create summary report
print("\n5. Creating summary report...")
summary = {
    "export_timestamp": timestamp,
    "graph_stats": stats,
    "layers": {
        "bronze": len([d for d in all_docs if d["texts"][1].endswith("bronze")]),
        "silver": len([d for d in all_docs if d["texts"][1].endswith("silver")]),
        "gold": len([d for d in all_docs if d["texts"][1].endswith("gold")]),
        "unknown": len([d for d in all_docs if d["texts"][1].endswith("unknown")]),
    },
    "namespaces": list(set(d["texts"][2].split(": ")[1] for d in all_docs)),
    "ready_for_rag": True,
}

with open(f"{OUTPUT_DIR}/summary_report_{timestamp}.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ Saved summary_report.json")

print("\n" + "=" * 80)
print(f"✓ ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
print("=" * 80)

print("\nSummary:")
print(f"  Total datasets: {stats['total_nodes']}")
print(f"  Total edges: {stats['total_edges']}")
print(f"  Bronze: {summary['layers']['bronze']}")
print(f"  Silver: {summary['layers']['silver']}")
print(f"  Gold: {summary['layers']['gold']}")
print(f"  Is DAG: {stats['is_dag']}")
print("\nFiles created:")
print(f"  - full_graph_{timestamp}.json")
print(f"  - rag_docs_*_{timestamp}.json (5 files)")
print(f"  - graph_stats_{timestamp}.json")
print(f"  - summary_report_{timestamp}.json")
