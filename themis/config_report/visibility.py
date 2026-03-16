"""Verbosity-aware filtering for configuration reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from themis.config_report.models import (
    ConfigReportDocument,
    ConfigReportHeader,
    ConfigReportNode,
    ConfigReportParameter,
)
from themis.config_report.types import ConfigReportVerbosity

_MEANINGFUL_FIELD_NAMES = {
    "project_name",
    "researcher_id",
    "model_id",
    "provider",
    "task_id",
    "dataset_id",
    "source",
    "split",
    "revision",
    "messages",
    "id",
    "metrics",
    "transform",
    "name",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop_sequences",
    "response_format",
    "seed",
    "kind",
    "count",
    "strata_field",
    "item_ids",
    "metadata_filters",
}

_INFRASTRUCTURE_CLASS_NAMES = {
    "SqliteBlobStorageSpec",
    "PostgresBlobStorageSpec",
    "_StorageSpecBase",
    "ExecutionPolicySpec",
    "LocalExecutionBackendSpec",
    "WorkerPoolExecutionBackendSpec",
    "BatchExecutionBackendSpec",
}


def _explicit_default_visibility(parameter: ConfigReportParameter) -> bool | None:
    return parameter.default_visibility


def _is_non_empty_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _is_non_empty_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
    )


def _node_is_infrastructure(node: ConfigReportNode) -> bool:
    return node.class_name in _INFRASTRUCTURE_CLASS_NAMES


def _has_meaningful_extras(parameter: ConfigReportParameter) -> bool:
    return _is_non_empty_mapping(parameter.value) or _is_non_empty_sequence(
        parameter.value
    )


def _is_meaningful_named_parameter(
    node: ConfigReportNode,
    parameter: ConfigReportParameter,
) -> bool | None:
    if parameter.name == "schema_version":
        return False
    if _node_is_infrastructure(node):
        return False
    if parameter.name in _MEANINGFUL_FIELD_NAMES:
        return True
    if parameter.name == "extras" and (_has_meaningful_extras(parameter)):
        return True
    if parameter.name == "metadata":
        return False
    return None


def _is_meaningful_value_change(parameter: ConfigReportParameter) -> bool:
    if not parameter.has_default:
        return True
    if parameter.value != parameter.default:
        if parameter.name in {"metadata", "extras"} and not _has_meaningful_extras(
            parameter
        ):
            return False
        return True
    return False


def _parameter_is_meaningful(
    node: ConfigReportNode,
    parameter: ConfigReportParameter,
) -> bool:
    explicit_visibility = _explicit_default_visibility(parameter)
    if explicit_visibility is not None:
        return explicit_visibility

    named_parameter_result = _is_meaningful_named_parameter(node, parameter)
    if named_parameter_result is not None:
        return named_parameter_result

    return _is_meaningful_value_change(parameter)


def _filter_node(
    node: ConfigReportNode,
    *,
    verbosity: str,
    keep_root: bool = False,
) -> ConfigReportNode | None:
    if verbosity == "full":
        return node

    filtered_children = [
        filtered_child
        for child in node.children
        if (filtered_child := _filter_node(child, verbosity=verbosity)) is not None
    ]
    filtered_parameters = [
        parameter
        for parameter in node.parameters
        if _parameter_is_meaningful(node, parameter)
    ]

    if not keep_root and not filtered_parameters and not filtered_children:
        return None

    return node.model_copy(
        update={"parameters": filtered_parameters, "children": filtered_children}
    )


def apply_verbosity(
    document: ConfigReportDocument,
    *,
    verbosity: ConfigReportVerbosity | str,
) -> ConfigReportDocument:
    """Return a verbosity-filtered view of a config report document."""

    if verbosity not in {"default", "full"}:
        raise ValueError(
            f"Unsupported config report verbosity '{verbosity}'. "
            "Expected one of: default, full."
        )

    header: ConfigReportHeader = document.header.model_copy(
        update={"verbosity": verbosity}
    )
    if verbosity == "full":
        return document.model_copy(update={"header": header})

    root = _filter_node(document.root, verbosity=verbosity, keep_root=True)
    assert root is not None
    return document.model_copy(update={"header": header, "root": root})
