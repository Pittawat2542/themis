"""Server commands."""

from __future__ import annotations

import sys
from typing import Annotated

from cyclopts import Parameter

from themis.cli.utils import resolve_storage_root
from themis.utils import logging_utils


def serve_command(
    *,
    port: Annotated[int, Parameter(help="Port to run server on")] = 8080,
    host: Annotated[str, Parameter(help="Host to bind to")] = "127.0.0.1",
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    reload: Annotated[bool, Parameter(help="Enable auto-reload (dev mode)")] = False,
    verbose: Annotated[bool, Parameter(help="Enable debug logging")] = False,
    json_logs: Annotated[bool, Parameter(help="Output logs as JSON")] = False,
) -> int:
    """Start the Themis API server with REST and WebSocket endpoints."""
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )
    try:
        from themis.server import create_app
        import uvicorn
    except ImportError:
        print("Error: FastAPI server dependencies not installed", file=sys.stderr)
        print("\nInstall with: pip install themis[server]", file=sys.stderr)
        print("           or: uv pip install themis[server]", file=sys.stderr)
        return 1

    # Determine storage path
    storage_path = resolve_storage_root(storage)

    print("Starting Themis API server...")
    print(f"  URL:     http://{host}:{port}")
    print(f"  Storage: {storage_path}")
    print(f"  Docs:    http://{host}:{port}/docs")
    print()

    # Create app
    app_instance = create_app(storage_path=storage_path)

    # Run server
    try:
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except Exception:
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Server failed", exc_info=True)
        return 1

    return 0
