# src/rag/index.py
from typing import Any, Dict, List

import faiss
import numpy as np

# import faiss
from sentence_transformers import SentenceTransformer


class RAGIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []
        self.emb: np.ndarray | None = None

    def build(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        texts = [list(d.values())[0] for d in docs]
        emb = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        self.emb = emb
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self.index = index

    def search(self, query: str, topk: int = 8) -> List[Dict[str, Any]]:
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, topk)

        return [self.docs[i] for i in I[0]]
