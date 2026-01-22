"""FastAPI application for Themis server.

This module defines the main FastAPI app with REST endpoints and WebSocket support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest
from themis.experiment.storage import ExperimentStorage


class RunSummary(BaseModel):
    """Summary of an experiment run."""
    
    run_id: str
    experiment_id: str = "default"
    status: str
    num_samples: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: str | None = None


class RunDetail(BaseModel):
    """Detailed information about a run."""
    
    run_id: str
    experiment_id: str = "default"
    status: str
    num_samples: int
    metrics: Dict[str, float]
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    """Request to compare multiple runs."""
    
    run_ids: List[str]
    metrics: List[str] | None = None
    statistical_test: str = "bootstrap"
    alpha: float = 0.05


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: str | None = None


def create_app(storage_path: str | Path = ".cache/experiments") -> FastAPI:
    """Create FastAPI application.
    
    Args:
        storage_path: Path to experiment storage
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Themis API",
        description="REST API for Themis experiment management",
        version="2.0.0",
    )
    
    # Enable CORS for web dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize storage
    storage = ExperimentStorage(storage_path)
    
    # Mount static files (dashboard)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="static")
    
    # WebSocket connection manager
    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []
        
        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
        
        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)
        
        async def broadcast(self, message: dict):
            for connection in self.active_connections:
                await connection.send_json(message)
    
    manager = ConnectionManager()
    
    # ===== REST ENDPOINTS =====
    
    @app.get("/", tags=["health"])
    async def root():
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "themis-api",
            "version": "2.0.0",
        }
    
    @app.get("/api/runs", response_model=List[RunSummary], tags=["runs"])
    async def list_runs():
        """List all experiment runs."""
        run_ids = storage.list_runs()
        
        summaries = []
        for run_id in run_ids:
            # Load basic info
            eval_records = storage.load_cached_evaluations(run_id)
            
            # Calculate average metrics
            metrics_dict: Dict[str, List[float]] = {}
            for record in eval_records.values():
                for metric_name, score_obj in record.scores.items():
                    if metric_name not in metrics_dict:
                        metrics_dict[metric_name] = []
                    
                    # Extract numeric score
                    if hasattr(score_obj, 'value'):
                        metrics_dict[metric_name].append(score_obj.value)
                    elif isinstance(score_obj, (int, float)):
                        metrics_dict[metric_name].append(float(score_obj))
            
            # Average metrics
            avg_metrics = {
                name: sum(scores) / len(scores) if scores else 0.0
                for name, scores in metrics_dict.items()
            }
            
            summaries.append(RunSummary(
                run_id=run_id,
                experiment_id="default",
                status="completed",
                num_samples=len(eval_records),
                metrics=avg_metrics,
            ))
        
        return summaries
    
    @app.get("/api/runs/{run_id}", response_model=RunDetail, tags=["runs"])
    async def get_run(run_id: str):
        """Get detailed information about a run."""
        if run_id not in storage.list_runs():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        # Load records
        eval_records = storage.load_cached_evaluations(run_id)
        gen_records_dict = storage.load_cached_records(run_id)
        
        # Calculate metrics
        metrics_dict: Dict[str, List[float]] = {}
        samples = []
        
        for cache_key, eval_record in eval_records.items():
            # Get generation record
            gen_record = gen_records_dict.get(cache_key)
            
            # Extract scores
            scores = {}
            for metric_name, score_obj in eval_record.scores.items():
                if hasattr(score_obj, 'value'):
                    value = score_obj.value
                elif isinstance(score_obj, (int, float)):
                    value = float(score_obj)
                else:
                    continue
                
                scores[metric_name] = value
                
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(value)
            
            # Build sample
            sample = {
                "id": gen_record.id if gen_record else cache_key,
                "prompt": gen_record.prompt if gen_record else "",
                "response": gen_record.response if gen_record else "",
                "scores": scores,
            }
            samples.append(sample)
        
        # Average metrics
        avg_metrics = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in metrics_dict.items()
        }
        
        return RunDetail(
            run_id=run_id,
            experiment_id="default",
            status="completed",
            num_samples=len(eval_records),
            metrics=avg_metrics,
            samples=samples,
        )
    
    @app.delete("/api/runs/{run_id}", tags=["runs"])
    async def delete_run(run_id: str):
        """Delete a run."""
        if run_id not in storage.list_runs():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        # Note: Current storage doesn't implement delete
        # This is a placeholder for future implementation
        raise HTTPException(
            status_code=501,
            detail="Delete not implemented in current storage"
        )
    
    @app.post("/api/compare", tags=["comparison"])
    async def compare_runs_api(request: ComparisonRequest):
        """Compare multiple runs."""
        # Validate runs exist
        existing_runs = set(storage.list_runs())
        for run_id in request.run_ids:
            if run_id not in existing_runs:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run not found: {run_id}"
                )
        
        if len(request.run_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 runs to compare"
            )
        
        # Parse statistical test
        try:
            test_enum = StatisticalTest(request.statistical_test)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid statistical test: {request.statistical_test}"
            )
        
        # Run comparison
        report = compare_runs(
            run_ids=request.run_ids,
            storage_path=storage._base_dir,
            metrics=request.metrics,
            statistical_test=test_enum,
            alpha=request.alpha,
        )
        
        return report.to_dict()
    
    @app.get("/api/benchmarks", tags=["presets"])
    async def list_benchmarks():
        """List available benchmark presets."""
        from themis.presets import list_benchmarks
        
        benchmarks = list_benchmarks()
        return {"benchmarks": benchmarks}
    
    # ===== WEBSOCKET ENDPOINTS =====
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates.
        
        Messages sent from server:
        - {"type": "run_started", "run_id": "...", "data": {...}}
        - {"type": "run_progress", "run_id": "...", "progress": 0.5}
        - {"type": "run_completed", "run_id": "...", "data": {...}}
        - {"type": "error", "message": "..."}
        
        Messages expected from client:
        - {"type": "subscribe", "run_id": "..."}
        - {"type": "unsubscribe", "run_id": "..."}
        - {"type": "ping"}
        """
        await manager.connect(websocket)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif msg_type == "subscribe":
                    run_id = message.get("run_id")
                    # TODO: Implement run subscription logic
                    await websocket.send_json({
                        "type": "subscribed",
                        "run_id": run_id
                    })
                
                elif msg_type == "unsubscribe":
                    run_id = message.get("run_id")
                    # TODO: Implement unsubscribe logic
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "run_id": run_id
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
        
        except WebSocketDisconnect:
            manager.disconnect(websocket)
    
    return app


__all__ = ["create_app"]
