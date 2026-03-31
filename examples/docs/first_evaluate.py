from __future__ import annotations

from themis import evaluate
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Run the smallest end-to-end evaluation through the Layer 1 API."""

    result = evaluate(
        model="builtin/demo_generator",
        data=[
            Dataset(
                dataset_id="sample",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
    )
    return {"run_id": result.run_id, "status": result.status.value}


if __name__ == "__main__":
    print(run_example())
