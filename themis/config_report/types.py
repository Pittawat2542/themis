"""Public type aliases for config-report formats and verbosity levels."""

from __future__ import annotations

from typing import Literal, TypeAlias

ConfigReportFormat: TypeAlias = Literal["json", "yaml", "markdown", "latex"]
ConfigReportVerbosity: TypeAlias = Literal["default", "full"]
