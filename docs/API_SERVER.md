# Themis API Server

The Themis API server provides REST and WebSocket endpoints for accessing experiment results, comparing runs, and monitoring experiments in real-time.

## Installation

The API server requires optional dependencies:

```bash
pip install themis[server]
# or
uv pip install themis[server]
```

## Quick Start

### Start the Server

```bash
# Default (port 8080, localhost)
themis serve

# Custom port and storage
themis serve --port 3000 --storage ~/.themis/runs

# Development mode with auto-reload
themis serve --reload --host 0.0.0.0
```

### Programmatic Usage

```python
from themis.server import create_app
import uvicorn

# Create app
app = create_app(storage_path=".cache/experiments")

# Run server
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## API Endpoints

### Health Check

**GET** `/`

Check if the server is running.

```bash
curl http://localhost:8080/
```

Response:
```json
{
  "status": "ok",
  "service": "themis-api",
  "version": "2.0.0"
}
```

### List Runs

**GET** `/api/runs`

List all experiment runs with summary metrics.

```bash
curl http://localhost:8080/api/runs
```

Response:
```json
[
  {
    "run_id": "run-2024-01-15",
    "experiment_id": "default",
    "status": "completed",
    "num_samples": 100,
    "metrics": {
      "ExactMatch": 0.85,
      "BLEU": 0.72
    },
    "created_at": null
  }
]
```

### Get Run Details

**GET** `/api/runs/{run_id}`

Get detailed information about a specific run, including all samples.

```bash
curl http://localhost:8080/api/runs/my-run-id
```

Response:
```json
{
  "run_id": "my-run-id",
  "experiment_id": "default",
  "status": "completed",
  "num_samples": 100,
  "metrics": {
    "ExactMatch": 0.85
  },
  "samples": [
    {
      "id": "sample-1",
      "prompt": "What is 2+2?",
      "response": "4",
      "scores": {
        "ExactMatch": 1.0
      }
    }
  ],
  "metadata": {}
}
```

### Compare Runs

**POST** `/api/compare`

Compare multiple runs with statistical significance testing.

Request body:
```json
{
  "run_ids": ["run-1", "run-2"],
  "metrics": ["ExactMatch", "BLEU"],
  "statistical_test": "bootstrap",
  "alpha": 0.05
}
```

```bash
curl -X POST http://localhost:8080/api/compare \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["run-1", "run-2"], "statistical_test": "bootstrap"}'
```

Response:
```json
{
  "run_ids": ["run-1", "run-2"],
  "metrics": ["ExactMatch"],
  "best_run_per_metric": {
    "ExactMatch": "run-1"
  },
  "overall_best_run": "run-1",
  "pairwise_results": [
    {
      "metric": "ExactMatch",
      "run_a": "run-1",
      "run_b": "run-2",
      "run_a_mean": 0.85,
      "run_b_mean": 0.80,
      "delta": 0.05,
      "delta_percent": 6.25,
      "winner": "run-1",
      "significant": true,
      "p_value": 0.001
    }
  ]
}
```

### List Benchmarks

**GET** `/api/benchmarks`

List available benchmark presets.

```bash
curl http://localhost:8080/api/benchmarks
```

Response:
```json
{
  "benchmarks": [
    "demo",
    "math500",
    "gsm8k",
    "aime24",
    "mmlu_pro",
    "supergpqa"
  ]
}
```

## WebSocket API

### Connection

Connect to the WebSocket endpoint:

```
ws://localhost:8080/ws
```

### Messages from Client

#### Ping

```json
{"type": "ping"}
```

Response:
```json
{"type": "pong"}
```

#### Subscribe to Run

```json
{
  "type": "subscribe",
  "run_id": "my-run-id"
}
```

Response:
```json
{
  "type": "subscribed",
  "run_id": "my-run-id"
}
```

#### Unsubscribe

```json
{
  "type": "unsubscribe",
  "run_id": "my-run-id"
}
```

### Messages from Server

#### Run Started

```json
{
  "type": "run_started",
  "run_id": "my-run-id",
  "data": {...}
}
```

#### Run Progress

```json
{
  "type": "run_progress",
  "run_id": "my-run-id",
  "progress": 0.5
}
```

#### Run Completed

```json
{
  "type": "run_completed",
  "run_id": "my-run-id",
  "data": {...}
}
```

#### Error

```json
{
  "type": "error",
  "message": "Error description"
}
```

## Python Client Examples

### REST API Client

```python
import requests

base_url = "http://localhost:8080"

# List runs
response = requests.get(f"{base_url}/api/runs")
runs = response.json()

# Get run details
run_id = runs[0]['run_id']
response = requests.get(f"{base_url}/api/runs/{run_id}")
detail = response.json()

# Compare runs
response = requests.post(
    f"{base_url}/api/compare",
    json={
        "run_ids": ["run-1", "run-2"],
        "statistical_test": "bootstrap",
        "alpha": 0.05
    }
)
comparison = response.json()
```

### WebSocket Client

```python
import asyncio
import json
import websockets

async def monitor_runs():
    uri = "ws://localhost:8080/ws"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to run
        await websocket.send(json.dumps({
            "type": "subscribe",
            "run_id": "my-run"
        }))
        
        # Listen for updates
        async for message in websocket:
            data = json.loads(message)
            print(f"Update: {data}")

asyncio.run(monitor_runs())
```

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

These interfaces allow you to:
- Browse all available endpoints
- See request/response schemas
- Try out API calls directly from the browser

## CORS Configuration

By default, the server allows requests from any origin (CORS `*`). For production, configure appropriate origins:

```python
from themis.server.app import create_app

app = create_app(storage_path=".cache/experiments")

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Authentication

The default server has no authentication. For production use, consider adding:

- API key authentication
- JWT tokens
- OAuth2 integration

Example with API key middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Use in endpoints
@app.get("/api/runs", dependencies=[Depends(get_api_key)])
async def list_runs():
    ...
```

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install Themis with server extras
RUN pip install themis[server]

# Expose port
EXPOSE 8080

# Run server
CMD ["themis", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### systemd Service

```ini
[Unit]
Description=Themis API Server
After=network.target

[Service]
Type=simple
User=themis
WorkingDirectory=/home/themis
ExecStart=/usr/local/bin/themis serve --host 0.0.0.0 --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name themis.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8080/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8080

# Use a different port
themis serve --port 8081
```

### Import Errors

If you get `ImportError: cannot import name 'create_app'`:

```bash
# Install server dependencies
pip install themis[server]
```

### CORS Issues

If requests from your web app are blocked:

1. Check browser console for CORS errors
2. Verify `allow_origins` in CORS middleware
3. Ensure you're using the correct protocol (http/https)

## Performance Tips

1. **Use gunicorn/uvicorn workers**: For production, run with multiple workers
   ```bash
   gunicorn themis.server.app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Enable HTTP/2**: Use a reverse proxy (nginx, caddy) with HTTP/2 support

3. **Add caching**: Cache frequent queries (list runs, etc.)

4. **Limit response sizes**: Paginate large result sets

5. **Use connection pooling**: For database backends

## Next Steps

- Build a web dashboard using the API (HTML/JavaScript or React)
- Implement real-time run monitoring via WebSocket
- Add authentication and authorization
- Deploy to production with proper security
