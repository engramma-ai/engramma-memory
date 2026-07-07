"""
Engramma Memory - Building a Chatbot with Long-Term Memory

This example shows how to give your chatbot persistent, compositional memory.
The bot remembers past conversations and can blend related memories together.
"""
import numpy as np
from engramma_memory import EngrammaMemory


def simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Simple deterministic embedding (use a real model in production)."""
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


class ChatbotWithMemory:
    def __init__(self):
        self.mem = EngrammaMemory(dim=128, backend="local")
        self.conversations: dict = {}
        self.turn_count = 0

    def remember(self, text: str):
        """Store a message in memory."""
        embedding = simple_embed(text)
        self.mem.store(key=embedding, value=embedding)
        self.conversations[self.turn_count] = text
        self.turn_count += 1

    def recall(self, query: str, top_k: int = 3) -> list:
        """Recall relevant past messages."""
        if self.mem.count == 0:
            return []

        embedding = simple_embed(query)
        results = self.mem.query(embedding, top_k=top_k)

        recalled = []
        for r in results:
            if r["score"] > 0.2:
                best_text = self._find_closest(r["value"])
                if best_text:
                    recalled.append(best_text)
        return recalled

    def compose_context(self, topics: list) -> str:
        """
        Blend multiple topics into a unified context.
        This is what makes Engramma special vs. plain vector search.
        """
        embeddings = [simple_embed(t) for t in topics]
        composed = self.mem.compose(embeddings)
        best = self._find_closest(composed)
        return best or ""

    def _find_closest(self, value: np.ndarray) -> str:
        best_dist = float('inf')
        best_text = ""
        for idx, text in self.conversations.items():
            emb = simple_embed(text)
            dist = float(np.linalg.norm(emb - value.flatten()[:128]))
            if dist < best_dist:
                best_dist = dist
                best_text = text
        return best_text


if __name__ == "__main__":
    bot = ChatbotWithMemory()

    # Simulate a conversation
    messages = [
        "I love programming in Python",
        "Machine learning is fascinating",
        "I'm building a recommendation system",
        "The weather is nice today",
        "I need help with PyTorch",
        "My project deadline is next Friday",
    ]

    print("=== Storing conversation ===")
    for msg in messages:
        bot.remember(msg)
        print(f"  Stored: '{msg}'")

    print("\n=== Recalling relevant memories ===")
    query = "What do I know about AI projects?"
    recalled = bot.recall(query, top_k=3)
    print(f"  Query: '{query}'")
    for r in recalled:
        print(f"  -> {r}")

    print("\n=== Composing context from multiple topics ===")
    context = bot.compose_context(["Python", "machine learning"])
    print(f"  Topics: ['Python', 'machine learning']")
    print(f"  Composed context: '{context}'")

    print(f"\n=== Stats ===")
    print(f"  {bot.mem.stats()}")
