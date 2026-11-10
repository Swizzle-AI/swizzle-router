from typing import List
import numpy as np

class MatryoshkaEmbedding:
    """
    Matryoshka Representation Learning (MRL) slice helper.
    Enables multi-tiered search:
      - 64-dim: Fast pruning & approximate filtering (<0.2ms)
      - 256-dim: Candidate reranking (<0.8ms)
      - 1024-dim: Full precision semantic verification
    """
    SUPPORTED_DIMS = [64, 128, 256, 512, 1024]

    @staticmethod
    def slice_and_normalize(embedding: np.ndarray, target_dim: int) -> np.ndarray:
        if target_dim not in MatryoshkaEmbedding.SUPPORTED_DIMS and target_dim > len(embedding):
            target_dim = len(embedding)
        sliced = embedding[:target_dim]
        norm = np.linalg.norm(sliced)
        return sliced / (norm + 1e-9)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9))
