"""Workflow-runner support types for Phase 3 evaluation."""

from __future__ import annotations


class WorkflowBuildError(ValueError):
    """Raised when a metric cannot build a valid evaluation workflow."""
