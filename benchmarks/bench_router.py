import time
from swizzle_router.router import SemanticRouter, Route
from swizzle_router.cache_affinity import CacheAffinityRouter, WorkerNode
from swizzle_router.intent import IntentClassifier
from swizzle_router.matryoshka import MatryoshkaEmbedding
import numpy as np

def run_benchmarks():
    print("=" * 80)
    print("SWIZZLE ROUTER LATENCY & MATRYOSHKA EMBEDDING BENCHMARK")
    print("=" * 80)

    router = SemanticRouter([
        Route(name="code", target_cluster="gpu-a100", sample_queries=["write python function", "fix bug in rust"]),
        Route(name="robotics", target_cluster="gpu-rtx4090", sample_queries=["move arm to coordinates", "compute quintic spline"]),
        Route(name="chat", target_cluster="gpu-general", sample_queries=["how are you", "tell me a story"])
    ])

    # 1. Routing Latency
    queries = [
        "def quicksort(arr):",
        "calculate end_effector trajectory delta",
        "write a novel about mars",
        "optimize SQL query performance",
    ]
    t0 = time.perf_counter()
    n = 1000
    for _ in range(n):
        for q in queries:
            _ = router.match(q)
    total_queries = n * len(queries)
    dur = time.perf_counter() - t0
    qps = total_queries / dur
    lat_us = (dur / total_queries) * 1e6
    print(f"1. Semantic Routing Throughput:    {qps:.0f} queries/sec")
    print(f"2. Mean Routing Latency:          {lat_us:.2f} microseconds (< 1.5 ms target)")

    # 2. Matryoshka Slicing Latency
    vec = np.random.randn(1024).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = MatryoshkaEmbedding.slice_and_normalize(vec, 64)
    slice_us = ((time.perf_counter() - t0) / 10000.0) * 1e6
    print(f"3. Matryoshka 64d Slice Latency:  {slice_us:.2f} microseconds")

    # 3. Intent Classification
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = IntentClassifier.classify("def binary_search(arr, x):")
    intent_us = ((time.perf_counter() - t0) / 10000.0) * 1e6
    print(f"4. Intent Classification Latency: {intent_us:.2f} microseconds")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
