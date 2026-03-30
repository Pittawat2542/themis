Inspect after the run:

- the SQLite file exists under `runs/`
- `get_execution_state(...)` returns the persisted run state
- `Reporter.export_markdown(...)` gives you a portable report without re-running the experiment
