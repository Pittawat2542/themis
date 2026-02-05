from __future__ import annotations

import multiprocessing

import pytest

from themis.evaluation.metrics.code.execution import ExecutionAccuracy


def _run_infinite_loop_case(result_queue):
    metric = ExecutionAccuracy(timeout=0.2)
    score = metric.compute(
        prediction=(
            "def solution(x):\n"
            "    while True:\n"
            "        pass\n"
        ),
        references=[
            {
                "function_name": "solution",
                "inputs": [1],
                "expected": [1],
            }
        ],
    )
    result_queue.put(score.details["results"][0])


def test_execution_accuracy_enforces_timeout():
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_infinite_loop_case, args=(queue,))
    process.start()
    process.join(timeout=3.0)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        pytest.fail("ExecutionAccuracy.compute hung instead of timing out")

    result = queue.get(timeout=1.0)
    assert result["status"] == "timeout"
    assert result["passed"] is False


def test_execution_accuracy_blocks_imports():
    metric = ExecutionAccuracy(timeout=0.5)
    score = metric.compute(
        prediction="import os\n\ndef solution(x):\n    return x\n",
        references=[
            {
                "function_name": "solution",
                "inputs": [1],
                "expected": [1],
            }
        ],
    )
    result = score.details["results"][0]
    assert result["status"] == "error"
    assert "import" in (result["error"] or "").lower()


def test_execution_accuracy_blocks_file_io():
    metric = ExecutionAccuracy(timeout=0.5)
    score = metric.compute(
        prediction=(
            "def solution(x):\n"
            "    open('/tmp/themis-exec-test.txt', 'w').write('x')\n"
            "    return x\n"
        ),
        references=[
            {
                "function_name": "solution",
                "inputs": [1],
                "expected": [1],
            }
        ],
    )
    result = score.details["results"][0]
    assert result["status"] == "error"
    assert "open" in (result["error"] or "").lower()


def test_execution_accuracy_blocks_network_access():
    metric = ExecutionAccuracy(timeout=0.5)
    score = metric.compute(
        prediction=(
            "def solution(x):\n"
            "    sock_mod = __import__('socket')\n"
            "    s = sock_mod.socket()\n"
            "    return x\n"
        ),
        references=[
            {
                "function_name": "solution",
                "inputs": [1],
                "expected": [1],
            }
        ],
    )
    result = score.details["results"][0]
    assert result["status"] == "error"
    assert "import" in (result["error"] or "").lower()


def test_execution_accuracy_blocks_process_spawn():
    metric = ExecutionAccuracy(timeout=0.5)
    score = metric.compute(
        prediction=(
            "def solution(x):\n"
            "    sub = __import__('subprocess')\n"
            "    sub.run(['echo', 'hi'])\n"
            "    return x\n"
        ),
        references=[
            {
                "function_name": "solution",
                "inputs": [1],
                "expected": [1],
            }
        ],
    )
    result = score.details["results"][0]
    assert result["status"] == "error"
    assert "import" in (result["error"] or "").lower()


def test_execution_accuracy_passes_simple_solution():
    metric = ExecutionAccuracy(timeout=0.5)
    score = metric.compute(
        prediction="def solution(a, b):\n    return a + b\n",
        references=[
            {
                "function_name": "solution",
                "inputs": [(1, 2), (3, 4)],
                "expected": [3, 7],
            }
        ],
    )
    assert score.value == 1.0
    assert score.details["results"][0]["status"] == "passed"
    assert score.details["results"][1]["status"] == "passed"
