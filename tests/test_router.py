import unittest
from swizzle_router.router import SemanticRouter, Route
from swizzle_router.cache_affinity import CacheAffinityRouter, WorkerNode
from swizzle_router.intent import IntentClassifier

class TestRouter(unittest.TestCase):
    def test_semantic_match(self):
        r = SemanticRouter([
            Route(name="code", target_cluster="gpu-a100", sample_queries=["write python function", "fix bug in rust"]),
            Route(name="robotics", target_cluster="gpu-rtx4090", sample_queries=["move arm to coordinates", "compute quintic spline"]),
        ])
        match = r.match("write python function to invert binary tree")
        self.assertEqual(match.route_name, "code")
        self.assertEqual(match.target_cluster, "gpu-a100")
        self.assertLess(match.latency_ms, 5.0)

    def test_cache_affinity(self):
        w1 = WorkerNode("node-1", "http://node-1:8000")
        w2 = WorkerNode("node-2", "http://node-2:8000")
        car = CacheAffinityRouter([w1, w2])
        chosen = car.select_best_worker("system prompt: you are an expert coder...")
        self.assertIn(chosen, [w1, w2])

    def test_intent_classification(self):
        self.assertEqual(IntentClassifier.classify("def fibonacci(n):"), "CODE_REASONING")
        self.assertEqual(IntentClassifier.classify("calculate end_effector trajectory"), "ROBOTICS_TELEOP")
        self.assertEqual(IntentClassifier.classify("hello there"), "CONVERSATION")

if __name__ == '__main__':
    unittest.main()
