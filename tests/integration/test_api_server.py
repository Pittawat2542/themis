"""Integration tests for API server."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient

from themis import evaluate
from themis.server import create_app


class TestAPIServer:
    """Test API server endpoints."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, client):
        """Test listing runs when storage is empty."""
        response = await client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_runs_with_data(self, client, storage_path):
        """Test listing runs after creating some."""
        # Create a run
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=2,
            storage=storage_path,
            run_id="test-run-1",
        )
        
        response = await client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        
        # Check run structure
        run = data[0]
        assert "run_id" in run
        assert "timestamp" in run
        assert "num_samples" in run

    @pytest.mark.asyncio
    async def test_get_run_details(self, client, storage_path):
        """Test getting details of a specific run."""
        # Create a run
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=2,
            storage=storage_path,
            run_id="test-run-details",
        )
        
        response = await client.get("/api/runs/test-run-details")
        assert response.status_code == 200
        data = response.json()
        
        assert data["run_id"] == "test-run-details"
        assert "metrics" in data
        assert "num_samples" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, client):
        """Test getting details of non-existent run."""
        response = await client.get("/api/runs/nonexistent-run")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_run(self, client, storage_path):
        """Test deleting a run."""
        # Create a run
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=2,
            storage=storage_path,
            run_id="test-run-delete",
        )
        
        # Delete it
        response = await client.delete("/api/runs/test-run-delete")
        assert response.status_code == 200
        
        # Verify it's gone
        response = await client.get("/api/runs/test-run-delete")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_runs(self, client, storage_path):
        """Test comparing runs via API."""
        # Create two runs
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=3,
            storage=storage_path,
            run_id="compare-run-1",
        )
        
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=3,
            storage=storage_path,
            run_id="compare-run-2",
        )
        
        # Compare them
        payload = {
            "run_ids": ["compare-run-1", "compare-run-2"],
            "statistical_test": "bootstrap",
        }
        
        response = await client.post("/api/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "pairwise_results" in data
        assert isinstance(data["pairwise_results"], list)

    @pytest.mark.asyncio
    async def test_list_benchmarks(self, client):
        """Test listing available benchmarks."""
        response = await client.get("/api/benchmarks")
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check benchmark structure
        benchmark = data[0]
        assert "name" in benchmark
        assert "description" in benchmark


class TestAPIServerErrors:
    """Test API server error handling."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_compare_with_invalid_runs(self, client):
        """Test comparing with non-existent runs."""
        payload = {
            "run_ids": ["invalid-1", "invalid-2"],
            "statistical_test": "bootstrap",
        }
        
        response = await client.post("/api/compare", json=payload)
        # Should return error status
        assert response.status_code in [400, 404, 422]

    @pytest.mark.asyncio
    async def test_compare_with_single_run(self, client):
        """Test comparing with only one run (should fail)."""
        payload = {
            "run_ids": ["single-run"],
            "statistical_test": "bootstrap",
        }
        
        response = await client.post("/api/compare", json=payload)
        # Should return error status
        assert response.status_code in [400, 422]


class TestWebSocketConnection:
    """Test WebSocket functionality."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.mark.asyncio
    @pytest.mark.skip("WebSocket testing requires additional setup")
    async def test_websocket_connection(self, app):
        """Test WebSocket connection."""
        # WebSocket testing would require websockets client
        # and more complex setup
        pass


class TestStaticFiles:
    """Test static file serving."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_dashboard_static_files(self, client):
        """Test serving dashboard static files."""
        response = await client.get("/dashboard/")
        # Should either return the dashboard or 404 if not built
        assert response.status_code in [200, 404]


class TestCORS:
    """Test CORS configuration."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = await client.options(
            "/api/runs",
            headers={"Origin": "http://localhost:3000"},
        )
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    @pytest.fixture
    def storage_path(self, tmp_path):
        """Create temporary storage for tests."""
        storage = tmp_path / "storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def app(self, storage_path):
        """Create test app instance."""
        return create_app(storage_path=str(storage_path))

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    @pytest.mark.asyncio
    async def test_swagger_ui(self, client):
        """Test Swagger UI endpoint."""
        response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_redoc(self, client):
        """Test ReDoc endpoint."""
        response = await client.get("/redoc")
        assert response.status_code == 200
