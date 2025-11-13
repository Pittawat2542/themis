# API Generation from Docstrings

Themis provides utilities to automatically generate REST APIs from Python functions using their docstrings and type hints. This feature leverages **FastAPI** to create production-ready APIs with automatic OpenAPI/Swagger documentation.

## Overview

The `themis.utils.api_generator` module inspects Python functions to:
- Extract type hints for request/response validation
- Parse docstrings for parameter descriptions
- Generate Pydantic models for request bodies
- Create FastAPI endpoints with proper routing
- Generate OpenAPI documentation automatically

## Installation

To use the API generation feature, install FastAPI and uvicorn:

```bash
# Add to your project
pip install fastapi uvicorn

# Or using uv
uv pip install fastapi uvicorn
```

## Quick Start

### Generate API from a Module

```python
from themis.utils.api_generator import create_api_from_module
from themis.evaluation import statistics

# Create API from all public functions in the statistics module
app = create_api_from_module(
    module=statistics,
    title="Themis Statistics API",
    prefix="/api/v1/stats"
)
```

### Generate API from Specific Functions

```python
from themis.utils.api_generator import create_api_from_functions
from themis.evaluation.statistics import (
    compute_confidence_interval,
    compute_statistical_summary,
)

app = create_api_from_functions(
    functions=[
        compute_confidence_interval,
        compute_statistical_summary,
    ],
    title="Statistics API",
    prefix="/api/stats"
)
```

### Run the API Server

Save your API in a file (e.g., `api_server.py`):

```python
from themis.utils.api_generator import create_api_from_module
from themis.evaluation import statistics

app = create_api_from_module(
    module=statistics,
    prefix="/api/stats"
)
```

Then run with uvicorn:

```bash
uvicorn api_server:app --reload --port 8000
```

Access the API:
- **API**: http://localhost:8000/api/stats/compute_confidence_interval
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Example Usage

### Create a Complete API Server

Create `themis_api.py`:

```python
"""Themis API Server - Auto-generated REST API for experiment utilities."""

from themis.utils.api_generator import create_api_from_module
from themis.evaluation import statistics
from themis.utils import cost_tracking

from fastapi import FastAPI

# Create main app
app = FastAPI(
    title="Themis Experiment API",
    description="REST API for Themis experiment utilities",
    version="1.0.0",
)

# Create sub-apps for different modules
stats_app = create_api_from_module(
    module=statistics,
    prefix="/stats",
)

# Mount sub-apps
app.mount("/api", stats_app)

@app.get("/")
def root():
    return {
        "message": "Themis API",
        "docs": "/docs",
        "endpoints": {
            "statistics": "/api/stats",
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the server:

```bash
python themis_api.py
```

### Make API Requests

```python
import requests

# Call the compute_confidence_interval endpoint
response = requests.post(
    "http://localhost:8000/api/stats/compute_confidence_interval",
    json={
        "values": [0.85, 0.82, 0.90, 0.88, 0.86],
        "confidence_level": 0.95
    }
)

result = response.json()
print(result)
# {
#   "result": {
#     "mean": 0.862,
#     "lower": 0.8415,
#     "upper": 0.8825,
#     "confidence_level": 0.95,
#     "sample_size": 5
#   }
# }
```

## How It Works

### 1. Function Introspection

The API generator inspects functions to extract:
- **Type hints**: For request validation and response schemas
- **Docstrings**: For API documentation and parameter descriptions
- **Default values**: For optional parameters

Example function:

```python
def compute_confidence_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute confidence interval for a sample mean.
    
    Args:
        values: Sequence of numeric values
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        ConfidenceInterval with bounds and statistics
    """
    # Implementation...
```

### 2. Request Model Generation

A Pydantic model is automatically created for the request body:

```python
class ComputeConfidenceIntervalRequest(BaseModel):
    values: Sequence[float]
    confidence_level: float = 0.95
```

### 3. Endpoint Registration

An endpoint is registered with:
- **Path**: `/api/stats/compute_confidence_interval` (based on function name)
- **Method**: POST
- **Request validation**: Using the generated Pydantic model
- **Response format**: `{"result": <function_return_value>}`
- **Documentation**: From docstring

### 4. OpenAPI Schema

FastAPI automatically generates OpenAPI (Swagger) documentation at `/docs`:

![API Docs Screenshot](https://via.placeholder.com/800x400?text=Interactive+API+Documentation)

## Advanced Usage

### Custom API Configuration

```python
from themis.utils.api_generator import create_api_from_module

app = create_api_from_module(
    module=my_module,
    title="Custom API",
    description="My custom API with detailed description",
    version="2.0.0",
    prefix="/api/v2",
    include_private=False,  # Exclude functions starting with _
)
```

### Adding Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "my-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

# Apply to all endpoints
app = create_api_from_module(
    module=statistics,
    prefix="/api/stats"
)

# Add authentication to specific routes
@app.middleware("http")
async def authenticate(request, call_next):
    # Add your authentication logic
    response = await call_next(request)
    return response
```

### Generate API Documentation

```python
from themis.utils.api_generator import (
    create_api_from_module,
    generate_api_documentation,
)

app = create_api_from_module(module=statistics)

# Generate markdown documentation
generate_api_documentation(
    app=app,
    output_path="docs/api_reference.md"
)
```

## Supported Docstring Formats

The API generator supports **Google-style docstrings**:

```python
def my_function(param1: str, param2: int = 10) -> dict:
    """Short description of the function.
    
    Longer description with more details about what the function does
    and how it should be used.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Dictionary containing the result
    
    Raises:
        ValueError: If param1 is empty
    """
    pass
```

## Best Practices

### 1. Use Type Hints

Always provide type hints for function parameters and return values:

```python
# Good
def process_data(values: List[float], threshold: float) -> Dict[str, Any]:
    pass

# Bad (no type hints)
def process_data(values, threshold):
    pass
```

### 2. Write Comprehensive Docstrings

Include parameter descriptions and examples:

```python
def analyze_results(
    scores: List[float],
    baseline: float = 0.5,
) -> AnalysisResult:
    """Analyze experiment results against a baseline.
    
    This function computes various statistical measures to compare
    the given scores against a baseline threshold.
    
    Args:
        scores: List of score values between 0 and 1
        baseline: Baseline threshold for comparison (default: 0.5)
    
    Returns:
        AnalysisResult object with statistical measures
    
    Example:
        >>> scores = [0.8, 0.75, 0.9]
        >>> result = analyze_results(scores, baseline=0.7)
        >>> print(result.mean)
        0.8167
    """
    pass
```

### 3. Use Pydantic Models for Complex Types

For complex data structures, use Pydantic models:

```python
from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    model_name: str
    temperature: float
    max_tokens: int

def run_experiment(config: ExperimentConfig) -> dict:
    """Run experiment with given configuration."""
    pass
```

### 4. Handle Errors Gracefully

The API generator wraps functions and returns HTTP 500 for exceptions. Add proper error handling in your functions:

```python
def compute_metric(values: List[float]) -> float:
    """Compute metric from values.
    
    Args:
        values: Non-empty list of values
    
    Raises:
        ValueError: If values is empty
    """
    if not values:
        raise ValueError("Values list cannot be empty")
    return sum(values) / len(values)
```

## Limitations

1. **POST Only**: Currently, all endpoints are POST methods
2. **Simple Types**: Complex nested types may not serialize perfectly
3. **Sync Only**: Async functions are not yet supported
4. **No Streaming**: Response streaming is not supported

## Alternative Tools

If you need more advanced API generation features, consider:

- **FastAPI directly**: Write endpoints manually for full control
- **Connexion**: OpenAPI-first Python framework
- **Flask-RESTX**: Flask extension with API generation
- **Django REST Framework**: Full-featured REST API framework

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI Specification](https://swagger.io/specification/)
