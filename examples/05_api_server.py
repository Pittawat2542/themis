"""Example: Using the Themis API Server

This example shows how to use the FastAPI server for accessing
experiment results via REST API and WebSocket.

Requirements:
    pip install themis[server]
    # or
    uv pip install themis[server]
"""

import asyncio
import json


def example_start_server():
    """Start the API server programmatically."""
    from themis.server import create_app
    import uvicorn

    # Create app with custom storage
    app = create_app(storage_path=".cache/experiments")

    # Run server
    print("Starting Themis API server on http://localhost:8080")
    print("API docs available at http://localhost:8080/docs")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8080)


def example_rest_api_client():
    """Example of using the REST API with requests."""
    import requests

    base_url = "http://localhost:8080"

    # Health check
    response = requests.get(f"{base_url}/")
    print("Health check:", response.json())
    print()

    # List all runs
    response = requests.get(f"{base_url}/api/runs")
    runs = response.json()
    print(f"Found {len(runs)} runs:")
    for run in runs:
        print(f"  - {run['run_id']}: {run['num_samples']} samples")
    print()

    # Get detailed run info
    if runs:
        run_id = runs[0]["run_id"]
        response = requests.get(f"{base_url}/api/runs/{run_id}")
        detail = response.json()
        print(f"Run {run_id} details:")
        print(f"  Samples: {detail['num_samples']}")
        print(f"  Metrics: {detail['metrics']}")
        print()

    # Compare runs
    if len(runs) >= 2:
        run_ids = [run["run_id"] for run in runs[:2]]
        response = requests.post(
            f"{base_url}/api/compare",
            json={"run_ids": run_ids, "statistical_test": "bootstrap", "alpha": 0.05},
        )
        comparison = response.json()
        print(f"Comparison of {run_ids}:")
        print(f"  Best run: {comparison['overall_best_run']}")
        print()


async def example_websocket_client():
    """Example of using WebSocket for real-time updates."""
    import websockets

    uri = "ws://localhost:8080/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        # Send ping
        await websocket.send(json.dumps({"type": "ping"}))
        response = await websocket.recv()
        print(f"Received: {response}")

        # Subscribe to a run
        await websocket.send(
            json.dumps({"type": "subscribe", "run_id": "test-run-123"})
        )
        response = await websocket.recv()
        print(f"Received: {response}")

        # Wait for updates (in real usage, this would be in a loop)
        # For now, just close after a moment
        await asyncio.sleep(1)


def example_curl_commands():
    """Show example curl commands for the API."""
    print("Example API Commands")
    print("=" * 60)
    print()

    print("# Health check")
    print("curl http://localhost:8080/")
    print()

    print("# List all runs")
    print("curl http://localhost:8080/api/runs")
    print()

    print("# Get run details")
    print("curl http://localhost:8080/api/runs/my-run-id")
    print()

    print("# Compare runs")
    print("curl -X POST http://localhost:8080/api/compare \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"run_ids": ["run-1", "run-2"], "statistical_test": "bootstrap"}\'')
    print()

    print("# List benchmarks")
    print("curl http://localhost:8080/api/benchmarks")
    print()

    print("# View API documentation")
    print("Open http://localhost:8080/docs in your browser")
    print()


if __name__ == "__main__":
    print("Themis API Server Examples")
    print("=" * 60)
    print()
    print("This script demonstrates how to use the Themis API server.")
    print()
    print("To start the server, run:")
    print("  themis serve")
    print()
    print("Or programmatically:")
    print("  from themis.server import create_app")
    print("  import uvicorn")
    print("  app = create_app()")
    print("  uvicorn.run(app, port=8080)")
    print()
    print("=" * 60)
    print()

    # Show curl examples
    example_curl_commands()

    print()
    print("Once the server is running, you can test these commands:")
    print()
    print("Python REST API client:")
    print("  python examples-simple/05_api_server.py")
    print()
    print("WebSocket client:")
    print("  # Requires: pip install websockets")
    print("  # Then run async example")
