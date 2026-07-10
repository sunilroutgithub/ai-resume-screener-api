from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once at import time — reused across requests instead of reloading per call.
# all-MiniLM-L6-v2 is a small, fast, free, locally-run embedding model (~80MB).
_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    """Convert a piece of text into a vector embedding."""
    return _model.encode(text, normalize_embeddings=True)


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts.
    Since embeddings are normalized, cosine similarity = dot product.
    Returns a float between 0.0 and 1.0.
    """
    embedding_a = get_embedding(text_a)
    embedding_b = get_embedding(text_b)
    similarity = float(np.dot(embedding_a, embedding_b))
    # Clamp in case of floating point drift slightly outside [0,1]
    return max(0.0, min(1.0, similarity))