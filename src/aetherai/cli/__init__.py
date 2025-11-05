import sys
from pathlib import Path

from aetherai.cli.demo_project import generate_demo_project
from aetherai.cli.legacy_ui import legacy_ui
from aetherai.cli.main import app
from aetherai.cli.migrate import migrate
from aetherai.cli.migrate import migrate_status
from aetherai.cli.migrate import run_migrations
from aetherai.cli.report import run_report
from aetherai.cli.ui import ui

__all__ = [
    "app",
    "ui",
    "legacy_ui",
    "generate_demo_project",
    "run_report",
    "migrate",
    "migrate_status",
    "run_migrations",
]

sys.path.append(str(Path.cwd()))
if __name__ == "__main__":
    app()
