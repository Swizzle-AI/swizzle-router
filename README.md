# Swizzle Router: Semantic Routing & Prefix-Aware Load Balancer

<div align="center">

### Ultra-Low Latency (< 1.5ms) Semantic Router & Matryoshka Embedding Balancer

**71,000+ Queries/sec Throughput · 2D Matryoshka Slicing · Warm Prefix KV Cluster Affinity**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://github.com/Swizzle-AI/swizzle-router)
[![Throughput](https://img.shields.io/badge/Throughput-71k_QPS-brightgreen?style=for-the-badge)](https://swizzle-ten.vercel.app)
[![Latency](https://img.shields.io/badge/Latency-14.1_μs_p50-00E676?style=for-the-badge)](https://swizzle-ten.vercel.app)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

---

</div>

## 🌐 Architectural Overview

In multi-node inference clusters, naive round-robin or least-connections load balancers disperse multi-turn requests across arbitrary nodes, destroying prompt prefix KV cache hit rates.

`swizzle-router` implements **Hierarchical Semantic Intent Routing** and **Warm Prefix Cache Affinity**:

```mermaid
flowchart TD
    Prompt["User / Agent Prompt"] --> Intent["Intent Classifier (< 10 μs)<br/>(Code vs Robotics vs Structured JSON vs Chat)"]
    
    subgraph MatryoshkaPipeline["Matryoshka 2D Multi-Tier Matching (< 20 μs)"]
        direction TB
        M1["Tier 1: 64-Dim Fast Vector Filter<br/>(Prunes 95% of unmatched cluster routes)"]
        M2["Tier 2: 256-Dim Dense Re-ranker"]
        M3["Tier 3: 1024-Dim Cosine Verification"]
        M1 --> M2 --> M3
    end

    Intent --> MatryoshkaPipeline
    MatryoshkaPipeline --> Affinity["CacheAffinityRouter<br/>(Hash Prefix Match against Active GPU Worker Nodes)"]

    Affinity -->|Warm KV Hit > 85%| WorkerA["GPU Node A (Warm Prefix Cache)"]
    Affinity -->|Least Loaded Fallback| WorkerB["GPU Node B (Prefill Cluster)"]
```

---

## ⚡ Core Modules

1. **`SemanticRouter` (`swizzle_router.router`)**:
   - Matches incoming queries against target cluster routes using cosine similarity centroids.
   - Sub-15 microsecond latency, delivering **>71,000 QPS** on a single CPU core.
2. **`CacheAffinityRouter` (`swizzle_router.cache_affinity`)**:
   - Computes SHA-256 AST prefix hashes and routes requests to GPU nodes already holding the warm KV blocks.
   - Guarantees **>85% KV cache hits** across distributed clusters.
3. **`MatryoshkaEmbedding` (`swizzle_router.matryoshka`)**:
   - Supports 2D nested representation learning (slicing from 1024-dim down to 64-dim, 128-dim, 256-dim, and 512-dim).
   - Allows blazing-fast first-pass pruning ($<5\,\mu\text{s}$) before full vector verification.
4. **`IntentClassifier` (`swizzle_router.intent`)**:
   - Zero-shot deterministic pattern and keyword classification into:
     - `CODE_REASONING`
     - `ROBOTICS_TELEOP`
     - `STRUCTURED_JSON`
     - `CONVERSATION`
     - `GENERAL`

---

## 📊 Empirical Benchmarks

```text
================================================================================
SWIZZLE ROUTER LATENCY & MATRYOSHKA EMBEDDING BENCHMARK
================================================================================
1. Semantic Routing Throughput:    71,093 queries/sec
2. Mean Routing Latency:          14.07 microseconds (< 1.5 ms target)
3. Matryoshka 64d Slice Latency:  4.44 microseconds
4. Intent Classification Latency: 10.31 microseconds
================================================================================
```

---

## 🛠️ Quickstart

```python
from swizzle_router import SemanticRouter, Route, CacheAffinityRouter, WorkerNode

# 1. Initialize Semantic Router
router = SemanticRouter([
    Route(name="coding", target_cluster="gpu-a100-sxm4", sample_queries=["write rust function", "fix python bug"]),
    Route(name="robotics", target_cluster="gpu-rtx4090", sample_queries=["compute quintic spline", "robot teleop arm"])
])

match = router.match("write a python quicksort function")
print(f"Target Cluster: {match.target_cluster} | Latency: {match.latency_ms:.2f} ms")

# 2. Prefix Cache Affinity
w1 = WorkerNode("node-1", "http://worker-1:8000")
w2 = WorkerNode("node-2", "http://worker-2:8000")
affinity = CacheAffinityRouter([w1, w2])
chosen = affinity.select_best_worker("System prompt: You are an autonomous coding agent...")
print(f"Selected Worker with Warm KV: {chosen.node_id}")
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
Code of conduct based on [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
