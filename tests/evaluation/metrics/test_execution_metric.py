from __future__ import annotations

import multiprocessing

import pytest

from themis.evaluation.metrics.code.execution import (
    ExecutionAccuracy,
    _execution_batch_worker,
    _execution_worker,
)


class _CaptureConn:
    def __init__(self):
        self.messages: list[dict[str, object]] = []
        self.closed = False

    def send(self, payload):
        self.messages.append(payload)

    def close(self):
        self.closed = True


def _run_infinite_loop_case(result_queue):
    metric = ExecutionAccuracy(timeout=0.2)
    score = metric.compute(
        prediction=("def solution(x):\n    while True:\n        pass\n"),
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


def test_execution_accuracy_enforces_memory_limit():
    metric = ExecutionAccuracy(timeout=1.0, max_memory_mb=32)
    score = metric.compute(
        prediction=("def solution(x):\n    data = [0] * 20000000\n    return x\n"),
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
    assert "memory" in (result["error"] or "").lower()


def test_execution_worker_success_payload_is_emitted():
    conn = _CaptureConn()
    _execution_worker(
        code="def solution(a, b):\n    return a + b\n",
        function_name="solution",
        test_input=(2, 3),
        expected_output=5,
        max_memory_mb=0,
        result_conn=conn,
    )

    assert conn.closed is True
    assert len(conn.messages) == 1
    payload = conn.messages[0]
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["output"] == "5"
    assert payload["error"] is None


def test_execution_worker_reports_missing_function():
    conn = _CaptureConn()
    _execution_worker(
        code="def not_solution(x):\n    return x\n",
        function_name="solution",
        test_input=1,
        expected_output=1,
        max_memory_mb=0,
        result_conn=conn,
    )

    payload = conn.messages[0]
    assert payload["status"] == "error"
    assert "not found" in str(payload["error"]).lower()


def test_execution_worker_reports_memory_error_message():
    conn = _CaptureConn()
    _execution_worker(
        code="def solution(x):\n    raise MemoryError()\n",
        function_name="solution",
        test_input=1,
        expected_output=1,
        max_memory_mb=0,
        result_conn=conn,
    )

    payload = conn.messages[0]
    assert payload["status"] == "error"
    assert "memory limit exceeded" in str(payload["error"]).lower()


def test_execution_batch_worker_uses_exception_class_when_message_empty():
    conn = _CaptureConn()
    _execution_batch_worker(
        code=(
            "def solution(x):\n"
            "    if x == 0:\n"
            "        raise ValueError()\n"
            "    return x\n"
        ),
        function_name="solution",
        test_inputs=[1, 0],
        expected_outputs=[1, 0],
        max_memory_mb=0,
        result_conn=conn,
    )

    payload = conn.messages[0]
    assert payload["status"] == "ok"
    assert payload["results"][0]["status"] == "passed"
    assert payload["results"][1]["status"] == "error"
    assert payload["results"][1]["error"] == "ValueError"


def test_execution_batch_worker_reports_missing_function():
    conn = _CaptureConn()
    _execution_batch_worker(
        code="def other(x):\n    return x\n",
        function_name="solution",
        test_inputs=[1],
        expected_outputs=[1],
        max_memory_mb=0,
        result_conn=conn,
    )

    payload = conn.messages[0]
    assert payload["status"] == "error"
    assert "not found" in str(payload["error"]).lower()
    assert payload["results"] == []


def test_execution_accuracy_execute_test_timeout_branch():
    metric = ExecutionAccuracy(timeout=0.05, max_memory_mb=0)
    result = metric._execute_test(
        ("def solution(x):\n    while True:\n        pass\n"),
        "solution",
        1,
        1,
    )
    assert result.status.value == "timeout"
    assert result.passed is False
