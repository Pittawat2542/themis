{
    "run_id": "{{project_name}}-001",
    "storage_dir": ".cache/{{project_name}}",
    "resume": true,
    "models": [
        {
            "name": "fake-model",
            "provider": "fake"
        }
    ],
    "samplings": [
        {
            "name": "greedy",
            "temperature": 0.0,
            "max_tokens": 512
        }
    ],
    "datasets": [
        {
            "name": "demo",
            "kind": "demo",
            "limit": 10
        }
    ]
}
