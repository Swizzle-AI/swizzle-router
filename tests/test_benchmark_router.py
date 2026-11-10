import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.bench_router import run_benchmarks


def test_router_benchmarks_run_cleanly():
    run_benchmarks()
