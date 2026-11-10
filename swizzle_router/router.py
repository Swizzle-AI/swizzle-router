import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np

@dataclass
class Route:
    name: str
    target_cluster: str
    sample_queries: List[str]
    threshold: float = 0.45
    metadata: Dict = field(default_factory=dict)

@dataclass
class RouteMatch:
    route_name: str
    target_cluster: str
    similarity: float
    latency_ms: float

class SemanticRouter:
    """
    Ultra-low latency (<2ms) semantic router mapping incoming user prompts
    to optimal inference clusters based on cosine similarity and prefix length.
    """
    def __init__(self, routes: Optional[List[Route]] = None, encoder: Optional[Callable] = None):
        self.routes = routes or []
        self.encoder = encoder or self._fallback_encoder
        self._route_vectors = {}
        if routes:
            self._precompute_routes()

    def add_route(self, route: Route):
        self.routes.append(route)
        self._precompute_routes()

    def _fallback_encoder(self, text: str) -> np.ndarray:
        """Deterministic 64-dim pseudo-embedding for fast CPU routing when torch is unmounted."""
        vec = np.zeros(64, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            h = abs(hash(w)) % 64
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def _precompute_routes(self):
        for r in self.routes:
            if not r.sample_queries:
                continue
            vecs = [self.encoder(q) for q in r.sample_queries]
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            self._route_vectors[r.name] = centroid / (norm + 1e-9)

    def match(self, prompt: str, default_cluster: str = "cluster-general") -> RouteMatch:
        start = time.perf_counter()
        q_vec = self.encoder(prompt)
        best_name = None
        best_sim = -1.0
        best_cluster = default_cluster

        for r in self.routes:
            r_vec = self._route_vectors.get(r.name)
            if r_vec is None:
                continue
            sim = float(np.dot(q_vec, r_vec))
            if sim > best_sim:
                best_sim = sim
                if sim >= r.threshold:
                    best_name = r.name
                    best_cluster = r.target_cluster

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return RouteMatch(
            route_name=best_name or "default",
            target_cluster=best_cluster,
            similarity=best_sim,
            latency_ms=elapsed_ms
        )
