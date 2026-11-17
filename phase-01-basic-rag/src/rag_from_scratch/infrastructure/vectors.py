"""Small vector helpers with no external dependencies."""

import math


def l2_normalize(vector: list[float]) -> list[float]:
    """Scale a vector to length 1.

    Gemini only normalizes its full-size (3072) embeddings. When we ask for a
    smaller size, the vectors must be normalized by us, as the docs recommend.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]
