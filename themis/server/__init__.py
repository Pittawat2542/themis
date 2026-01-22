"""FastAPI server for Themis web dashboard.

This module provides a REST API and WebSocket interface for:
- Listing and viewing experiment runs
- Comparing multiple runs
- Real-time monitoring of running experiments
- Exporting results in various formats

The server is optional and requires the 'server' extra:
    pip install themis[server]
    # or
    uv pip install themis[server]

Usage:
    # Start the server
    themis serve --port 8080
    
    # Or programmatically
    from themis.server import create_app
    app = create_app(storage_path=".cache/experiments")
    
    # Run with uvicorn
    uvicorn themis.server:app --host 0.0.0.0 --port 8080
"""

from themis.server.app import create_app

__all__ = ["create_app"]
