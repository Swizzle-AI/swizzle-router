import unittest
import numpy as np
from swizzle_router.matryoshka import MatryoshkaEmbedding

class TestMatryoshka(unittest.TestCase):
    def test_slicing_normalization(self):
        vec = np.random.randn(1024).astype(np.float32)
        v64 = MatryoshkaEmbedding.slice_and_normalize(vec, 64)
        self.assertEqual(len(v64), 64)
        self.assertAlmostEqual(float(np.linalg.norm(v64)), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
