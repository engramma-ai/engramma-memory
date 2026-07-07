"""
Engramma Memory - RAG Agent with Compositional Retrieval

This example demonstrates using Engramma for Retrieval-Augmented Generation
with a twist: compositional queries that blend multiple stored documents.

Unlike traditional RAG (retrieve top-k, concatenate), Engramma can natively
compose related documents through multi-head attention.
"""
import numpy as np
from engramma_memory import EngrammaMemory


def embed_document(text: str, dim: int = 256) -> np.ndarray:
    """Placeholder embedding function. Replace with your model."""
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


class EngrammaRAGAgent:
    """
    RAG agent with compositional retrieval powered by Engramma.

    Key advantages over standard RAG:
    - Native composition: "What connects topic A and topic B?" in one query
    - Smart routing: exact recall for known facts, composition for synthesis
    - Graceful degradation: noisy/partial queries still retrieve relevant docs
    """

    def __init__(self, dim: int = 256):
        self.mem = EngrammaMemory(dim=dim, backend="local")
        self.dim = dim
        self.documents: dict = {}
        self.doc_count = 0

    def ingest(self, documents: list):
        """Ingest documents into memory."""
        for doc in documents:
            embedding = embed_document(doc, self.dim)
            self.mem.store(key=embedding, value=embedding)
            self.documents[self.doc_count] = doc
            self.doc_count += 1

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Standard retrieval - find most relevant documents."""
        embedding = embed_document(query, self.dim)
        results = self.mem.query(embedding, top_k=top_k)

        docs = []
        for r in results:
            text = self._match_document(r["value"])
            if text:
                docs.append({"text": text, "score": r["score"]})
        return docs

    def retrieve_compositional(self, queries: list) -> dict:
        """
        Compositional retrieval: blend multiple query topics.

        This is Engramma's killer feature. Instead of retrieving separately
        for each query and concatenating, Engramma composes them through
        multi-head attention, finding documents that bridge ALL topics.
        """
        embeddings = [embed_document(q, self.dim) for q in queries]
        composed = self.mem.compose(embeddings)
        text = self._match_document(composed)
        return {
            "composed_result": text or "(synthesized across topics)",
            "queries": queries,
        }

    def answer(self, question: str, context_queries: list = None) -> dict:
        """
        Full RAG pipeline: retrieve context, then answer.

        If context_queries are provided, uses compositional retrieval
        to build richer context than simple top-k.
        """
        if context_queries:
            composition = self.retrieve_compositional(context_queries)
            context = composition["composed_result"]
        else:
            docs = self.retrieve(question, top_k=3)
            context = "\n".join(d["text"] for d in docs[:3])

        return {
            "question": question,
            "context": context,
            "answer": f"[LLM would answer here using context]\nContext used: {context[:100]}...",
        }

    def _match_document(self, value: np.ndarray) -> str:
        if not self.documents:
            return ""
        value = value.flatten()[:self.dim]
        best_dist = float('inf')
        best_text = ""
        for idx, text in self.documents.items():
            emb = embed_document(text, self.dim)
            dist = float(np.linalg.norm(emb - value))
            if dist < best_dist:
                best_dist = dist
                best_text = text
        return best_text


if __name__ == "__main__":
    agent = EngrammaRAGAgent(dim=256)

    # Ingest a knowledge base
    docs = [
        "Python is a high-level programming language known for its readability.",
        "Machine learning models require large datasets for training.",
        "Neural networks are inspired by biological brain structures.",
        "FastAPI is a modern Python web framework for building APIs.",
        "Vector databases store embeddings for similarity search.",
        "Engramma Memory combines exact recall with native composition.",
        "Transformers use self-attention mechanisms for sequence modeling.",
        "RAG combines retrieval with generation for grounded responses.",
    ]

    print("=== Ingesting documents ===")
    agent.ingest(docs)
    print(f"  Ingested {len(docs)} documents")
    print(f"  Memory stats: {agent.mem.stats()}")

    # Standard retrieval
    print("\n=== Standard Retrieval ===")
    results = agent.retrieve("How do neural networks work?", top_k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:60]}...")

    # Compositional retrieval (Engramma advantage)
    print("\n=== Compositional Retrieval (Engramma unique) ===")
    comp = agent.retrieve_compositional(["Python", "machine learning"])
    print(f"  Queries: {comp['queries']}")
    print(f"  Result: {comp['composed_result'][:80]}...")

    # Full RAG answer
    print("\n=== RAG Answer with Compositional Context ===")
    answer = agent.answer(
        "How can I build an ML API?",
        context_queries=["Python web framework", "machine learning", "API"]
    )
    print(f"  Question: {answer['question']}")
    print(f"  Context: {answer['context'][:80]}...")
