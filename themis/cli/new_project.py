from __future__ import annotations

from pathlib import Path


def create_project(project_name: str, project_path: Path) -> None:
    if (project_path / project_name).exists():
        raise FileExistsError(
            f"Project '{project_name}' already exists in {project_path}"
        )

    project_dir = project_path / project_name
    project_dir.mkdir()

    templates_dir = Path(__file__).parent / "templates"

    # Create config.sample.json
    with open(templates_dir / "config.sample.json.tpl", "r") as f:
        config_template = f.read()
    with open(project_dir / "config.sample.json", "w") as f:
        f.write(config_template.replace("{{project_name}}", project_name))

    # Create cli.py
    with open(templates_dir / "cli.py.tpl", "r") as f:
        cli_template = f.read()
    with open(project_dir / "cli.py", "w") as f:
        f.write(cli_template)

    # Create README.md
    with open(templates_dir / "README.md.tpl", "r") as f:
        readme_template = f.read()
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_template.replace("{{project_name}}", project_name))
