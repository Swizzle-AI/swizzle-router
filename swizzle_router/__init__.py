"""
Swizzle Router: Semantic Routing, Matryoshka Embeddings, and Prefix-Aware Cluster Load Balancing.
"""
from .router import SemanticRouter, RouteMatch
from .cache_affinity import CacheAffinityRouter, WorkerNode
from .matryoshka import MatryoshkaEmbedding
from .intent import IntentClassifier

__version__ = "0.1.0"
__all__ = [
    "SemanticRouter",
    "RouteMatch",
    "CacheAffinityRouter",
    "WorkerNode",
    "MatryoshkaEmbedding",
    "IntentClassifier",
]
