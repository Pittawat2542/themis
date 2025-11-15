"""API generation utilities for creating REST APIs from Python functions.

This module provides tools to automatically generate REST API endpoints from
Python functions using their docstrings and type hints. It leverages FastAPI
for automatic OpenAPI schema generation.

Example:
    ```python
    from themis.utils.api_generator import create_api_from_module

    # Generate API from a module
    app = create_api_from_module(
        module=themis.evaluation.statistics,
        prefix="/api/v1/statistics"
    )

    # Run the API server
    # uvicorn main:app --reload
    ```
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, get_type_hints

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, create_model

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    BaseModel = None
    create_model = None


class APIGenerationError(Exception):
    """Exception raised when API generation fails."""

    pass


def create_api_from_functions(
    functions: List[Callable],
    title: str = "Auto-Generated API",
    description: str = "API generated from Python functions",
    version: str = "1.0.0",
    prefix: str = "",
) -> Any:
    """Create a FastAPI application from a list of functions.

    This function inspects each function's signature, type hints, and docstring
    to automatically generate REST API endpoints with proper request/response
    validation and OpenAPI documentation.

    Args:
        functions: List of functions to expose as API endpoints
        title: API title
        description: API description
        version: API version
        prefix: URL prefix for all endpoints (e.g., "/api/v1")

    Returns:
        FastAPI application instance

    Raises:
        APIGenerationError: If FastAPI is not installed or function inspection fails

    Example:
        ```python
        from themis.evaluation.statistics import compute_confidence_interval

        app = create_api_from_functions(
            functions=[compute_confidence_interval],
            title="Statistics API",
            prefix="/api/stats"
        )
        ```
    """
    if not FASTAPI_AVAILABLE:
        raise APIGenerationError(
            "FastAPI is not installed. Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(title=title, description=description, version=version)

    for func in functions:
        _register_function_as_endpoint(app, func, prefix)

    return app


def create_api_from_module(
    module: Any,
    title: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    prefix: str = "",
    include_private: bool = False,
) -> Any:
    """Create a FastAPI application from all functions in a module.

    Args:
        module: Python module containing functions to expose
        title: API title (defaults to module name)
        description: API description (defaults to module docstring)
        version: API version
        prefix: URL prefix for all endpoints
        include_private: Whether to include private functions (starting with _)

    Returns:
        FastAPI application instance

    Raises:
        APIGenerationError: If FastAPI is not installed

    Example:
        ```python
        from themis.evaluation import statistics

        app = create_api_from_module(
            module=statistics,
            prefix="/api/stats"
        )
        ```
    """
    if not FASTAPI_AVAILABLE:
        raise APIGenerationError(
            "FastAPI is not installed. Install it with: pip install fastapi uvicorn"
        )

    # Extract module metadata
    if title is None:
        title = f"{module.__name__} API"

    if description is None:
        description = inspect.getdoc(module) or f"API for {module.__name__}"

    # Find all functions in the module
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip private functions unless explicitly included
        if not include_private and name.startswith("_"):
            continue

        # Only include functions defined in this module
        if obj.__module__ == module.__name__:
            functions.append(obj)

    return create_api_from_functions(
        functions=functions,
        title=title,
        description=description,
        version=version,
        prefix=prefix,
    )


def _register_function_as_endpoint(
    app: Any,
    func: Callable,
    prefix: str = "",
) -> None:
    """Register a single function as a POST endpoint in the FastAPI app.

    Args:
        app: FastAPI application instance
        func: Function to register
        prefix: URL prefix for the endpoint
    """
    func_name = func.__name__
    endpoint_path = f"{prefix}/{func_name}".replace("//", "/")

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Extract docstring
    docstring = inspect.getdoc(func) or f"Execute {func_name}"

    # Parse docstring to extract parameter descriptions
    param_docs = _parse_docstring_params(docstring)

    # Build Pydantic model for request body
    request_model = _create_request_model(func_name, sig, type_hints, param_docs)

    # Create endpoint function
    async def endpoint(request: request_model):  # type: ignore
        try:
            # Convert request model to dict
            params = request.dict()

            # Call the original function
            result = func(**params)

            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Set endpoint metadata
    endpoint.__name__ = f"endpoint_{func_name}"
    endpoint.__doc__ = docstring

    # Register the endpoint
    app.post(
        endpoint_path,
        response_model=Dict[str, Any],
        summary=f"Execute {func_name}",
        description=docstring,
    )(endpoint)


def _create_request_model(
    func_name: str,
    sig: inspect.Signature,
    type_hints: Dict[str, type],
    param_docs: Dict[str, str],
) -> type:
    """Create a Pydantic model for function parameters.

    Args:
        func_name: Function name (used for model name)
        sig: Function signature
        type_hints: Type hints dictionary
        param_docs: Parameter documentation from docstring

    Returns:
        Pydantic model class
    """
    fields = {}

    for param_name, param in sig.parameters.items():
        # Skip self/cls parameters
        if param_name in ("self", "cls"):
            continue

        # Get type hint or default to Any
        param_type = type_hints.get(param_name, Any)

        # Get default value
        if param.default is inspect.Parameter.empty:
            default = ...  # Required field
        else:
            default = param.default

        # Get description from docstring
        description = param_docs.get(param_name, "")

        # Create field with description
        fields[param_name] = (param_type, default)

    # Create model name
    model_name = f"{func_name.title().replace('_', '')}Request"

    # Create and return Pydantic model
    return create_model(model_name, **fields)


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Parse parameter descriptions from Google-style docstring.

    Args:
        docstring: Function docstring

    Returns:
        Dictionary mapping parameter names to descriptions
    """
    param_docs = {}

    if not docstring:
        return param_docs

    # Look for Args section
    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_desc = []

    for line in lines:
        stripped = line.strip()

        # Check if we're entering Args section
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue

        # Check if we're leaving Args section
        if in_args_section and stripped and not line.startswith(" "):
            break

        if in_args_section and stripped:
            # Check if this is a parameter line (has a colon)
            if ":" in stripped and not stripped.startswith(":"):
                # Save previous parameter
                if current_param:
                    param_docs[current_param] = " ".join(current_desc).strip()

                # Parse new parameter
                parts = stripped.split(":", 1)
                current_param = parts[0].strip()
                if len(parts) > 1:
                    current_desc = [parts[1].strip()]
                else:
                    current_desc = []
            elif current_param:
                # Continue description from previous line
                current_desc.append(stripped)

    # Save last parameter
    if current_param:
        param_docs[current_param] = " ".join(current_desc).strip()

    return param_docs


def generate_api_documentation(
    app: Any,
    output_path: str = "api_docs.md",
) -> None:
    """Generate markdown documentation for a FastAPI application.

    Args:
        app: FastAPI application instance
        output_path: Path to output markdown file
    """
    if not FASTAPI_AVAILABLE:
        raise APIGenerationError("FastAPI is not installed")

    lines = [
        f"# {app.title}",
        "",
        app.description,
        "",
        f"**Version:** {app.version}",
        "",
        "## Endpoints",
        "",
    ]

    for route in app.routes:
        if hasattr(route, "methods") and "POST" in route.methods:
            lines.append(f"### `POST {route.path}`")
            lines.append("")
            if route.description:
                lines.append(route.description)
                lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


__all__ = [
    "create_api_from_functions",
    "create_api_from_module",
    "generate_api_documentation",
    "APIGenerationError",
]


# Example usage
if __name__ == "__main__":
    # Check if FastAPI is available
    if not FASTAPI_AVAILABLE:
        print("FastAPI is not installed. Install with: pip install fastapi uvicorn")
        exit(1)

    # Example: Create API from evaluation.statistics module
    print("Example: Creating API from functions...")
    print("To use this utility:")
    print("1. Install FastAPI: pip install fastapi uvicorn")
    print("2. Create an API:")
    print("   from themis.utils.api_generator import create_api_from_module")
    print("   from themis.evaluation import statistics")
    print("   app = create_api_from_module(statistics, prefix='/api/stats')")
    print("3. Run the server:")
    print("   uvicorn your_module:app --reload")
