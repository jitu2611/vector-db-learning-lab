"""Inspect embeddings and perform semantic search without a vector database.

Run:
    pip install -r requirements.txt
    python embedding_demo.py

The first run downloads all-MiniLM-L6-v2 and caches it locally.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
SENTENCES = [
    "How do I deploy changes to production?",
    "How can I release my service to the live environment?",
    "How do I reset my account password?",
    "Production deployments require approval from the platform team.",
]


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of a vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A zero vector cannot be normalized.")
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: 1 means same direction; lower means less aligned."""
    return float(np.dot(normalize(a), normalize(b)))


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    # Each sentence becomes one 384-dimensional vector. These are raw vectors,
    # so their lengths may vary.
    embeddings = model.encode(SENTENCES, convert_to_numpy=True, normalize_embeddings=False)

    print(f"Model: {MODEL_NAME}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print("(4 sentences × 384 values per sentence)\n")

    for sentence, vector in zip(SENTENCES, embeddings):
        print(f"{sentence!r}")
        print(f"  first 8 values: {vector[:8]}")
        print(f"  raw vector length: {np.linalg.norm(vector):.4f}")

    print("\nPairwise cosine similarity:")
    pairs = [(0, 1), (0, 2), (0, 3)]
    for left, right in pairs:
        score = cosine_similarity(embeddings[left], embeddings[right])
        print(f"  {score:.4f}  |  {SENTENCES[left]!r}")
        print(f"          |  {SENTENCES[right]!r}")

    # This is brute-force semantic retrieval. A vector DB performs this same
    # idea efficiently and persistently for much larger collections.
    query = "Who needs to approve a release to production?"
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=False)
    scores = [
        (cosine_similarity(query_embedding, document_embedding), sentence)
        for sentence, document_embedding in zip(SENTENCES, embeddings)
    ]

    print(f"\nQuery: {query!r}")
    print("Ranked documents:")
    for score, sentence in sorted(scores, key=lambda result: result[0], reverse=True):
        print(f"  {score:.4f}  |  {sentence!r}")


if __name__ == "__main__":
    main()
