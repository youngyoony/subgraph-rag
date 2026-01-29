# Graph-Based Retrieval-Augmented Reasoning with Subgraph Context

## OVERVIEW
This project explores a graph-based Retrieval-Augmented Generation (RAG) framework that enhances large language model (LLM) reasoning by conditioning responses on minimal, query-relevant subgraphs rather than unstructured text alone.

Instead of treating complex analytical systems as flat documents, we represent them as directed graphs and retrieve structurally meaningful context for each question. The goal is to improve accuracy, explainability, and faithfulness in analytical question answering.

------------------------------------------------------------

## MOTIVATION
Standard RAG pipelines retrieve text passages independently, ignoring the structural relationships that define how analytical components interact.

In complex data systems, meaning is encoded in:
- dependencies between entities,
- transformation logic,
- and multi-step derivations.

This project investigates whether retrieving and reasoning over a minimal connecting subgraph leads to better analytical answers than document-level retrieval.

------------------------------------------------------------

## CORE IDEA
Given a natural language question:

1. Retrieve the most semantically relevant graph edges.
2. Extract the minimal directed subgraph that connects them.
3. Convert the subgraph into structured textual context.
4. Prompt an LLM using this graph-aware representation.
5. Compare answers with and without subgraph conditioning.

------------------------------------------------------------

## DATA REPRESENTATION
Analytical pipelines are encoded as directed graphs.

- Nodes represent entities such as tables, metrics, or features.
- Edges represent transformations or operations.
- Edge attributes contain transformation logic used for semantic retrieval.

Multiple datasets are supported through modular JSON inputs.

------------------------------------------------------------

## SYSTEM COMPONENTS

### Graph Construction
JSON files are parsed into directed graphs while preserving transformation-level semantics.

### Retrieval-Augmented Index
Each edge operation is indexed for semantic search.
For a given query, the system retrieves the top-k most relevant edges.

### Subgraph Extraction
A minimal directed subgraph connecting the retrieved edges is computed.
This preserves structural and causal relationships while removing irrelevant context.

### Graph-Aware Prompting
The extracted subgraph is converted into structured textual descriptions.
The LLM is prompted using this graph-aware context to encourage faithful reasoning.

------------------------------------------------------------

## KEY CONTRIBUTIONS
- Introduces subgraph-based retrieval for RAG systems
- Moves beyond document-level context to structural reasoning
- Improves interpretability and auditability of LLM outputs
- Reduces hallucination in analytical question answering

------------------------------------------------------------

## APPLICATIONS
- Complex analytical systems
- Feature and metric interpretability
- Risk modeling and stratification analysis
- Model governance and explainability
- High-stakes decision support systems

------------------------------------------------------------

## TAKEAWAY
By retrieving and reasoning over minimal, query-specific subgraphs, this project demonstrates a practical path toward more faithful and explainable LLM-assisted analysis in complex structured systems.


## Structure

```
├── poetry.lock                     
├── pyproject.toml                  
├── README.md                       
└── src
    ├── data_extraction             # Dataset construction and preprocessing
    │   ├── 1_cms_5datasets_integrated_new.ipynb
    │   ├── 2_cost_analytics_revised.ipynb
    │   ├── 3_geographic_health_equity.ipynb
    │   ├── 4_provider_performance_network.ipynb
    │   └── 5_payer_utilization_intelligence.ipynb
    ├── __init__.py                 
    ├── lineage                     # Graph construction and dependency modeling
    │   ├── 2_cost_analytics_revised.ipynb
    │   ├── 3_geographic_health_equity.ipynb
    │   ├── 4_provider_performance_network.ipynb
    │   ├── 5_payer_utilization_intelligence.ipynb
    │   ├── dag_service.py           # DAG-based graph construction utilities
    │   ├── docker-compose-enhanced.yml  
    │   ├── export_final_outputs.py  # Export finalized graph artifacts
    │   ├── fetch_table_jobs.py      # Fetch and manage upstream table jobs
    │   └── test_dag_service.py      # Tests for DAG service logic
    └── rag
        ├── index.py                 # RAG indexing logic
        ├── __init__.py              
        ├── prompt.py                # Prompt construction for LLM
        ├── rag_total.py             # End-to-end RAG pipeline
        ├── subgraph.py              # Subgraph extraction and visualization
        └── utils.py                 # Shared utilities
```