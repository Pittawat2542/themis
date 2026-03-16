"""Recursive collection of nested config objects into one report tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from datetime import datetime, timezone
from enum import Enum
import inspect
from pathlib import Path
import subprocess
from typing import Any, TypeGuard, cast, get_args, get_origin

from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from themis.config_report.metadata import get_config_report_options
from themis.config_report.models import (
    ConfigReportDocument,
    ConfigReportHeader,
    ConfigReportNode,
    ConfigReportParameter,
)
from themis.config_report.source_index import ClassSourceInfo, load_source_index
from themis.types.json_types import JSONValueType


def _best_effort_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _infer_entrypoint() -> str | None:
    for frame_info in inspect.stack()[2:]:
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            continue
        module_name = module.__name__
        if module_name.startswith("themis.config_report"):
            continue
        return f"{module_name}:{frame_info.function}"
    return None


def _display_type(annotation: object) -> str:
    if annotation is inspect._empty or annotation is None:
        return "unknown"
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation).replace("typing.", "")
    args = " | ".join(_display_type(arg) for arg in get_args(annotation))
    origin_name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    if origin_name == "UnionType":
        return args
    if origin_name in {"list", "set", "tuple"}:
        return f"{origin_name}[{args}]"
    if origin_name == "dict":
        return f"dict[{args}]"
    return str(annotation).replace("typing.", "")


def _is_scalar(value: object) -> bool:
    return isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
            Enum,
            Path,
            datetime,
            SecretStr,
        ),
    )


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_pydantic_model(value: object) -> TypeGuard[BaseModel]:
    return isinstance(value, BaseModel)


def _is_plain_object(value: object) -> bool:
    if _is_scalar(value) or _is_mapping(value) or _is_sequence(value):
        return False
    if dataclasses.is_dataclass(value):
        return True
    return hasattr(value, "__dict__") and not isinstance(value, type)


def _json_safe(value: object, *, redacted: bool = False) -> JSONValueType:
    if redacted:
        return "***REDACTED***"
    if isinstance(value, SecretStr):
        return "***REDACTED***"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if _is_scalar(item) or isinstance(item, (Mapping, Sequence))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _root_name(config: object) -> str:
    if _is_mapping(config) or _is_sequence(config):
        return "config"
    cls = config.__class__
    options = get_config_report_options(cls)
    return options.display_name or cls.__name__


def _project_name(config: object) -> str | None:
    if isinstance(config, Mapping):
        project = config.get("project")
        if project is not None and hasattr(project, "project_name"):
            value = getattr(project, "project_name")
            if isinstance(value, str):
                return value
    if hasattr(config, "project_name"):
        value = getattr(config, "project_name")
        if isinstance(value, str):
            return value
    return None


def _safe_default_factory(default_factory: object) -> JSONValueType | None:
    if not callable(default_factory):
        return None
    try:
        return _json_safe(default_factory())
    except Exception:
        return "<factory>"


def _pydantic_default(field_info: FieldInfo) -> tuple[JSONValueType | None, bool]:
    if field_info.default is not PydanticUndefined:
        return _json_safe(field_info.default), True
    if field_info.default_factory is not None:
        return _safe_default_factory(field_info.default_factory), True
    return None, False


def _dataclass_default(
    field: dataclasses.Field[object],
) -> tuple[JSONValueType | None, bool]:
    if field.default is not dataclasses.MISSING:
        return _json_safe(field.default), True
    if field.default_factory is not dataclasses.MISSING:
        return _safe_default_factory(field.default_factory), True
    return None, False


def _source_info_for_class(cls: type[object]) -> ClassSourceInfo | None:
    source_file = inspect.getsourcefile(cls)
    qualname = cls.__qualname__
    if source_file is None:
        return None
    try:
        return load_source_index(source_file).get_class_info(qualname)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _class_source_metadata(
    cls: type[object],
    *,
    class_info: ClassSourceInfo | None,
) -> tuple[str | None, int | None]:
    if class_info is not None:
        return class_info.source_file, class_info.source_line

    try:
        source_file = inspect.getsourcefile(cls)
    except (OSError, TypeError):
        source_file = None

    try:
        source_line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        source_line = None
    return source_file, source_line


def _declaring_class(cls: type[object], field_name: str) -> type[object]:
    for base in cls.__mro__:
        annotations = getattr(base, "__annotations__", {})
        if field_name in annotations:
            return base
    return cls


def _plain_object_fields(value: object) -> list[tuple[str, object, object]]:
    annotations = getattr(value.__class__, "__annotations__", {})
    names = list(annotations)
    for name in vars(value):
        if name not in names:
            names.append(name)
    fields: list[tuple[str, object, object]] = []
    for name in names:
        if name.startswith("_") or not hasattr(value, name):
            continue
        fields.append(
            (name, getattr(value, name), annotations.get(name, inspect._empty))
        )
    return fields


def _parameter(
    *,
    name: str,
    value: object,
    annotation: object,
    default: JSONValueType | None,
    has_default: bool,
    declared_in: str | None,
    class_info: ClassSourceInfo | None,
    doc: str | None,
    inline_comment: str | None,
    default_visibility: bool | None = None,
    redacted: bool = False,
) -> ConfigReportParameter:
    source_line = class_info.field_lines.get(name) if class_info is not None else None
    source_file = class_info.source_file if class_info is not None else None
    if inline_comment is None and class_info is not None:
        inline_comment = class_info.field_comments.get(name)
    return ConfigReportParameter(
        name=name,
        value=_json_safe(value, redacted=redacted),
        type_repr=_display_type(annotation),
        default=default,
        has_default=has_default,
        source_file=source_file,
        source_line=source_line,
        doc=doc,
        inline_comment=inline_comment,
        declared_in=declared_in,
        default_visibility=default_visibility,
    )


def _is_child_value(
    field_name: str,
    field_value: object,
    *,
    child_fields: set[str],
) -> bool:
    return (
        field_name in child_fields
        or _is_mapping(field_value)
        or _is_sequence(field_value)
        or _is_pydantic_model(field_value)
        or dataclasses.is_dataclass(field_value)
        or _is_plain_object(field_value)
    )


def _default_visibility(
    field_name: str,
    *,
    paper_fields: set[str],
    non_paper_fields: set[str],
) -> bool | None:
    if field_name in paper_fields:
        return True
    if field_name in non_paper_fields:
        return False
    return None


def _build_class_node(
    *,
    cls: type[object],
    class_info: ClassSourceInfo | None,
    name: str,
    path: str,
    depth: int,
    parent_path: str | None,
    docstring: str | None,
    parameters: list[ConfigReportParameter],
    children: list[ConfigReportNode],
) -> ConfigReportNode:
    source_file, source_line = _class_source_metadata(cls, class_info=class_info)
    return ConfigReportNode(
        name=name,
        path=path,
        depth=depth,
        parent_path=parent_path,
        class_name=cls.__name__,
        qualified_class_name=cls.__qualname__,
        module_name=cls.__module__,
        declared_in=cls.__name__,
        source_file=source_file,
        source_line=source_line,
        docstring=docstring,
        parameters=parameters,
        children=children,
    )


class ConfigReportCollector:
    """Collect a nested config-report document from supported config objects."""

    def __init__(self) -> None:
        self._active_ids: set[int] = set()

    def build_document(
        self,
        config: object,
        *,
        entrypoint: str | None = None,
    ) -> ConfigReportDocument:
        """Build the full config report document."""

        root = self._collect_node(
            config,
            name=_root_name(config),
            path="$",
            depth=0,
            parent_path=None,
        )
        header = ConfigReportHeader(
            generated_at=datetime.now(timezone.utc).isoformat(),
            git_commit=_best_effort_git_commit(),
            project_name=_project_name(config),
            entrypoint=entrypoint or _infer_entrypoint(),
            root_type=config.__class__.__name__,
        )
        return ConfigReportDocument(header=header, root=root)

    def _collect_node(
        self,
        value: object,
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        identity = id(value)
        if not _is_scalar(value) and identity in self._active_ids:
            return ConfigReportNode(
                name=name,
                path=path,
                depth=depth,
                parent_path=parent_path,
                class_name="cycle",
                qualified_class_name=None,
                module_name=None,
                declared_in=None,
                source_file=None,
                source_line=None,
                docstring="Cycle reference detected.",
                parameters=[
                    ConfigReportParameter(
                        name="cycle_ref",
                        value="cycle detected",
                        type_repr="str",
                    )
                ],
                children=[],
            )

        self._active_ids.add(identity)
        try:
            if _is_mapping(value):
                node = self._collect_mapping(
                    value, name=name, path=path, depth=depth, parent_path=parent_path
                )
            elif _is_sequence(value):
                node = self._collect_sequence(
                    value, name=name, path=path, depth=depth, parent_path=parent_path
                )
            elif _is_pydantic_model(value):
                node = self._collect_pydantic(
                    value, name=name, path=path, depth=depth, parent_path=parent_path
                )
            elif dataclasses.is_dataclass(value):
                node = self._collect_dataclass(
                    value, name=name, path=path, depth=depth, parent_path=parent_path
                )
            elif _is_plain_object(value):
                node = self._collect_plain_object(
                    value, name=name, path=path, depth=depth, parent_path=parent_path
                )
            else:
                node = ConfigReportNode(
                    name=name,
                    path=path,
                    depth=depth,
                    parent_path=parent_path,
                    class_name=value.__class__.__name__,
                    qualified_class_name=value.__class__.__qualname__,
                    module_name=value.__class__.__module__,
                    parameters=[
                        ConfigReportParameter(
                            name="value",
                            value=_json_safe(value),
                            type_repr=_display_type(type(value)),
                        )
                    ],
                    children=[],
                )
            return node
        finally:
            self._active_ids.discard(identity)

    def _collect_mapping(
        self,
        value: Mapping[object, object],
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        parameters: list[ConfigReportParameter] = []
        children: list[ConfigReportNode] = []
        for key, item in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path != "$" else f"$.{key_name}"
            if _is_scalar(item):
                parameters.append(
                    ConfigReportParameter(
                        name=key_name,
                        value=_json_safe(item),
                        type_repr=_display_type(type(item)),
                    )
                )
            else:
                children.append(
                    self._collect_node(
                        item,
                        name=key_name,
                        path=child_path,
                        depth=depth + 1,
                        parent_path=path,
                    )
                )
        return ConfigReportNode(
            name=name,
            path=path,
            depth=depth,
            parent_path=parent_path,
            class_name="mapping",
            qualified_class_name=None,
            module_name=None,
            declared_in=None,
            source_file=None,
            source_line=None,
            docstring=None,
            parameters=parameters,
            children=children,
        )

    def _collect_sequence(
        self,
        value: Sequence[object],
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        parameters: list[ConfigReportParameter] = []
        children: list[ConfigReportNode] = []
        for index, item in enumerate(value):
            item_name = f"[{index}]"
            item_path = f"{path}[{index}]"
            if _is_scalar(item):
                parameters.append(
                    ConfigReportParameter(
                        name=item_name,
                        value=_json_safe(item),
                        type_repr=_display_type(type(item)),
                    )
                )
            else:
                children.append(
                    self._collect_node(
                        item,
                        name=item_name,
                        path=item_path,
                        depth=depth + 1,
                        parent_path=path,
                    )
                )
        return ConfigReportNode(
            name=name,
            path=path,
            depth=depth,
            parent_path=parent_path,
            class_name="sequence",
            qualified_class_name=None,
            module_name=None,
            declared_in=None,
            source_file=None,
            source_line=None,
            docstring=None,
            parameters=parameters,
            children=children,
        )

    def _collect_pydantic(
        self,
        value: BaseModel,
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        cls = value.__class__
        options = get_config_report_options(cls)
        class_info = _source_info_for_class(cls)
        docstring = (
            class_info.docstring if class_info is not None else inspect.getdoc(cls)
        )
        parameters: list[ConfigReportParameter] = []
        children: list[ConfigReportNode] = []

        for field_name, field_info in cls.model_fields.items():
            if field_name in options.hidden_fields:
                continue
            field_value = getattr(value, field_name)
            owner = _declaring_class(cls, field_name)
            owner_info = _source_info_for_class(owner)
            default, has_default = _pydantic_default(field_info)
            is_child = _is_child_value(
                field_name,
                field_value,
                child_fields=options.child_fields,
            )
            child_path = f"{path}.{field_name}" if path != "$" else f"$.{field_name}"
            if is_child:
                children.append(
                    self._collect_node(
                        field_value,
                        name=field_name,
                        path=child_path,
                        depth=depth + 1,
                        parent_path=path,
                    )
                )
                continue
            parameters.append(
                _parameter(
                    name=field_name,
                    value=field_value,
                    annotation=field_info.annotation,
                    default=default,
                    has_default=has_default,
                    declared_in=owner.__name__,
                    class_info=owner_info or class_info,
                    doc=field_info.description,
                    inline_comment=None,
                    default_visibility=_default_visibility(
                        field_name,
                        paper_fields=options.paper_fields,
                        non_paper_fields=options.non_paper_fields,
                    ),
                    redacted=field_name in options.redacted_fields
                    or _looks_secret(field_name),
                )
            )

        return _build_class_node(
            cls=cls,
            class_info=class_info,
            name=options.display_name or name,
            path=path,
            depth=depth,
            parent_path=parent_path,
            docstring=docstring,
            parameters=parameters,
            children=children,
        )

    def _collect_dataclass(
        self,
        value: object,
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        cls = value.__class__
        options = get_config_report_options(cls)
        class_info = _source_info_for_class(cls)
        docstring = (
            class_info.docstring if class_info is not None else inspect.getdoc(cls)
        )
        parameters: list[ConfigReportParameter] = []
        children: list[ConfigReportNode] = []

        for field in dataclasses.fields(cast(Any, value)):
            if field.name in options.hidden_fields:
                continue
            field_value = getattr(value, field.name)
            owner = _declaring_class(cls, field.name)
            owner_info = _source_info_for_class(owner)
            default, has_default = _dataclass_default(field)
            is_child = _is_child_value(
                field.name,
                field_value,
                child_fields=options.child_fields,
            )
            child_path = f"{path}.{field.name}" if path != "$" else f"$.{field.name}"
            if is_child:
                children.append(
                    self._collect_node(
                        field_value,
                        name=field.name,
                        path=child_path,
                        depth=depth + 1,
                        parent_path=path,
                    )
                )
                continue
            parameters.append(
                _parameter(
                    name=field.name,
                    value=field_value,
                    annotation=field.type,
                    default=default,
                    has_default=has_default,
                    declared_in=owner.__name__,
                    class_info=owner_info or class_info,
                    doc=None,
                    inline_comment=None,
                    default_visibility=_default_visibility(
                        field.name,
                        paper_fields=options.paper_fields,
                        non_paper_fields=options.non_paper_fields,
                    ),
                    redacted=field.name in options.redacted_fields
                    or _looks_secret(field.name),
                )
            )

        return _build_class_node(
            cls=cls,
            class_info=class_info,
            name=options.display_name or name,
            path=path,
            depth=depth,
            parent_path=parent_path,
            docstring=docstring,
            parameters=parameters,
            children=children,
        )

    def _collect_plain_object(
        self,
        value: object,
        *,
        name: str,
        path: str,
        depth: int,
        parent_path: str | None,
    ) -> ConfigReportNode:
        cls = value.__class__
        options = get_config_report_options(cls)
        class_info = _source_info_for_class(cls)
        docstring = (
            class_info.docstring if class_info is not None else inspect.getdoc(cls)
        )
        parameters: list[ConfigReportParameter] = []
        children: list[ConfigReportNode] = []

        for field_name, field_value, annotation in _plain_object_fields(value):
            if field_name in options.hidden_fields:
                continue
            owner = _declaring_class(cls, field_name)
            owner_info = _source_info_for_class(owner)
            is_child = _is_child_value(
                field_name,
                field_value,
                child_fields=options.child_fields,
            )
            child_path = f"{path}.{field_name}" if path != "$" else f"$.{field_name}"
            if is_child:
                children.append(
                    self._collect_node(
                        field_value,
                        name=field_name,
                        path=child_path,
                        depth=depth + 1,
                        parent_path=path,
                    )
                )
                continue
            parameters.append(
                _parameter(
                    name=field_name,
                    value=field_value,
                    annotation=annotation,
                    default=None,
                    has_default=False,
                    declared_in=owner.__name__,
                    class_info=owner_info or class_info,
                    doc=None,
                    inline_comment=None,
                    default_visibility=_default_visibility(
                        field_name,
                        paper_fields=options.paper_fields,
                        non_paper_fields=options.non_paper_fields,
                    ),
                    redacted=field_name in options.redacted_fields
                    or _looks_secret(field_name),
                )
            )

        return _build_class_node(
            cls=cls,
            class_info=class_info,
            name=options.display_name or name,
            path=path,
            depth=depth,
            parent_path=parent_path,
            docstring=docstring,
            parameters=parameters,
            children=children,
        )


def _looks_secret(field_name: str) -> bool:
    lowered = field_name.lower()
    if "secret" in lowered or "password" in lowered:
        return True
    if "api_key" in lowered or "apikey" in lowered:
        return True
    tokens = set(lowered.replace("-", "_").split("_"))
    return "token" in tokens


def build_config_report_document(
    config: object,
    *,
    entrypoint: str | None = None,
) -> ConfigReportDocument:
    """Collect one nested config report document."""

    return ConfigReportCollector().build_document(config, entrypoint=entrypoint)
