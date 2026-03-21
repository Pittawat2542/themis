"""Typed registry for engines, metrics, extractors, judges, and hooks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import inspect
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from themis.contracts.protocols import (
    Extractor,
    InferenceEngine,
    JudgeService,
    Metric,
    PipelineHook,
)
from themis.errors import SpecValidationError
from themis.specs.experiment import ToolHandler, TrialSpec
from themis.types.enums import ErrorCode, ResponseFormat


SUPPORTED_PLUGIN_API_MAJOR = 1
_PluginT = TypeVar("_PluginT")


def _default_response_formats() -> set[ResponseFormat]:
    return {ResponseFormat.TEXT}


class EngineCapabilities(BaseModel):
    """Declared execution-time capabilities for an inference engine plugin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    supports_response_format: set[ResponseFormat] = Field(
        default_factory=_default_response_formats
    )
    supports_logprobs: bool = False
    supports_seed: bool = Field(
        default=False,
        description=(
            "Whether the engine honours the ``seed`` field in InferenceParamsSpec. "
            "When False, deterministic seeding relies solely on Themis-level candidate "
            "seed derivation; the engine may still produce non-deterministic outputs."
        ),
    )
    max_context_tokens: int | None = None


PromptTokenEstimator = Callable[[TrialSpec], int]


@dataclass(frozen=True, slots=True)
class PluginRegistration(Generic[_PluginT]):
    """Metadata describing one registered plugin entry."""

    name: str
    factory: Callable[[], _PluginT] | type[_PluginT] | _PluginT
    version: str
    plugin_api: str
    registration_order: int


@dataclass(frozen=True, slots=True)
class InferenceEngineRegistration(PluginRegistration[InferenceEngine]):
    """Inference-engine registration plus declared capability metadata."""

    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    prompt_token_estimator: PromptTokenEstimator | None = None


@dataclass(frozen=True, slots=True)
class HookRegistration:
    """Ordered pipeline hook registration."""

    name: str
    hook: PipelineHook
    priority: int = 100
    idempotent: bool = True
    registration_order: int = 0


class PluginRegistry:
    """
    Instance-scoped registry for protocol implementations and plugin metadata.
    """

    _KNOWN_MAPPING_KEYS: frozenset[str] = frozenset(
        {"engines", "metrics", "extractors", "judges", "hooks", "tools"}
    )

    def __init__(self) -> None:
        self._registration_order = 0
        self._inference_engines: dict[str, InferenceEngineRegistration] = {}
        self._extractors: dict[str, PluginRegistration[Extractor]] = {}
        self._metrics: dict[str, PluginRegistration[Metric]] = {}
        self._judges: dict[str, PluginRegistration[JudgeService]] = {}
        self._tools: dict[str, PluginRegistration[ToolHandler]] = {}
        self._hooks: list[HookRegistration] = []
        self._register_builtin_extractors()

    @classmethod
    def from_dict(cls, mapping: dict[str, dict[str, object]]) -> "PluginRegistry":
        """Construct a registry from a plain dict, reducing per-run registration boilerplate.

        Each top-level key selects a plugin category; the value is a ``{name: factory}``
        mapping.  Built-in extractors are always registered regardless of what ``mapping``
        contains.

        Supported keys:
          - ``engines``   — inference engine factories
          - ``metrics``   — metric factories
          - ``extractors`` — extractor factories (supplements built-ins)
          - ``judges``    — judge service factories
          - ``tools``     — opaque tool handler factories
          - ``hooks``     — ``PipelineHook`` instances (registered in iteration order)

        Example::

            registry = PluginRegistry.from_dict({
                "engines":  {"openai": OpenAIEngine},
                "metrics":  {"exact_match": ExactMatchMetric},
                "extractors": {"my_parser": MyExtractor},
            })
        """
        unknown = set(mapping) - cls._KNOWN_MAPPING_KEYS
        if unknown:
            raise ValueError(
                f"PluginRegistry.from_dict received unknown mapping key(s): "
                f"{sorted(unknown)}. Supported keys: {sorted(cls._KNOWN_MAPPING_KEYS)}."
            )
        registry = cls()
        for name, factory in mapping.get("engines", {}).items():
            registry.register_inference_engine(name, factory)  # type: ignore[arg-type]
        for name, factory in mapping.get("metrics", {}).items():
            registry.register_metric(name, factory)  # type: ignore[arg-type]
        for name, factory in mapping.get("extractors", {}).items():
            registry.register_extractor(name, factory)  # type: ignore[arg-type]
        for name, factory in mapping.get("judges", {}).items():
            registry.register_judge(name, factory)  # type: ignore[arg-type]
        for name, factory in mapping.get("tools", {}).items():
            registry.register_tool(name, factory)
        for name, hook in mapping.get("hooks", {}).items():
            registry.register_hook(name, hook)  # type: ignore[arg-type]
        return registry

    def register_inference_engine(
        self,
        name: str,
        factory: Callable[[], InferenceEngine]
        | type[InferenceEngine]
        | InferenceEngine,
        *,
        version: str = "0.0.0",
        capabilities: EngineCapabilities | None = None,
        plugin_api: str = "1.0",
        prompt_token_estimator: PromptTokenEstimator | None = None,
    ) -> None:
        """Register an inference engine implementation under a provider name."""
        self._inference_engines[name] = InferenceEngineRegistration(
            name=name,
            factory=factory,
            version=version,
            plugin_api=plugin_api,
            registration_order=self._next_registration_order(),
            capabilities=capabilities or EngineCapabilities(),
            prompt_token_estimator=prompt_token_estimator,
        )

    def register_extractor(
        self,
        name: str,
        factory: Callable[[], Extractor] | type[Extractor] | Extractor,
        *,
        version: str = "0.0.0",
        plugin_api: str = "1.0",
    ) -> None:
        """Register an extractor implementation."""
        self._extractors[name] = PluginRegistration(
            name=name,
            factory=factory,
            version=version,
            plugin_api=plugin_api,
            registration_order=self._next_registration_order(),
        )

    def register_metric(
        self,
        name: str,
        factory: Callable[[], Metric] | type[Metric] | Metric,
        *,
        version: str = "0.0.0",
        plugin_api: str = "1.0",
    ) -> None:
        """Register a metric implementation."""
        self._metrics[name] = PluginRegistration(
            name=name,
            factory=factory,
            version=version,
            plugin_api=plugin_api,
            registration_order=self._next_registration_order(),
        )

    def register_judge(
        self,
        name: str,
        factory: Callable[[], JudgeService] | type[JudgeService] | JudgeService,
        *,
        version: str = "0.0.0",
        plugin_api: str = "1.0",
    ) -> None:
        """Register a judge service implementation."""
        self._judges[name] = PluginRegistration(
            name=name,
            factory=factory,
            version=version,
            plugin_api=plugin_api,
            registration_order=self._next_registration_order(),
        )

    def register_tool(
        self,
        name: str,
        factory: Callable[[], ToolHandler] | type[ToolHandler] | ToolHandler,
        *,
        version: str = "0.0.0",
        plugin_api: str = "1.0",
    ) -> None:
        """Register an opaque runtime tool handler."""
        self._tools[name] = PluginRegistration(
            name=name,
            factory=factory,
            version=version,
            plugin_api=plugin_api,
            registration_order=self._next_registration_order(),
        )

    def register_hook(
        self,
        name: str,
        hook: PipelineHook,
        *,
        priority: int = 100,
        idempotent: bool = True,
    ) -> None:
        """Register a pipeline hook and preserve deterministic ordering metadata."""
        if not isinstance(hook, PipelineHook):
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=(
                    "Hook registrations must implement the full PipelineHook contract."
                ),
            )
        self._hooks.append(
            HookRegistration(
                name=name,
                hook=hook,
                priority=priority,
                idempotent=idempotent,
                registration_order=self._next_registration_order(),
            )
        )

    def has_inference_engine(self, name: str) -> bool:
        """Return whether an inference engine exists for ``name``."""
        return name in self._inference_engines

    def has_extractor(self, name: str) -> bool:
        """Return whether an extractor exists for ``name``."""
        return name in self._extractors

    def has_metric(self, name: str) -> bool:
        """Return whether a metric exists for ``name``."""
        return name in self._metrics

    def has_judge(self, name: str) -> bool:
        """Return whether a judge service exists for ``name``."""
        return name in self._judges

    def has_tool(self, name: str) -> bool:
        """Return whether an opaque runtime tool handler exists for ``name``."""
        return name in self._tools

    def has_hook(self, name: str) -> bool:
        """Return whether a pipeline hook with ``name`` is registered."""
        return any(h.name == name for h in self._hooks)

    def get_inference_engine_registration(
        self, name: str
    ) -> InferenceEngineRegistration:
        """Fetch inference-engine registration metadata for ``name``."""
        if name not in self._inference_engines:
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=f"Provider {name} not found in registry.",
            )
        return self._inference_engines[name]

    def get_extractor_registration(self, name: str) -> PluginRegistration[Extractor]:
        """Fetch extractor registration metadata for ``name``."""
        if name not in self._extractors:
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=f"Extractor {name} not found in registry.",
            )
        return self._extractors[name]

    def get_metric_registration(self, name: str) -> PluginRegistration[Metric]:
        """Fetch metric registration metadata for ``name``."""
        if name not in self._metrics:
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=f"Metric {name} not found in registry.",
            )
        return self._metrics[name]

    def get_tool_registration(self, name: str) -> PluginRegistration[ToolHandler]:
        """Fetch tool registration metadata for ``name``."""
        if name not in self._tools:
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=f"Tool {name} not found in registry.",
            )
        return self._tools[name]

    def get_inference_engine(self, name: str) -> InferenceEngine:
        """Instantiate or return the registered inference engine for ``name``."""
        registration = self.get_inference_engine_registration(name)
        return self._instantiate(registration.factory, required_methods=("infer",))

    def get_extractor(self, name: str) -> Extractor:
        """Instantiate or return the registered extractor for ``name``."""
        registration = self.get_extractor_registration(name)
        extractor = self._instantiate(
            registration.factory,
            required_methods=("extract",),
        )
        self._validate_extractor_signature(name, extractor)
        return extractor

    def get_metric(self, name: str) -> Metric:
        """Instantiate or return the registered metric for ``name``."""
        registration = self.get_metric_registration(name)
        return self._instantiate(registration.factory, required_methods=("score",))

    def get_judge(self, name: str) -> JudgeService:
        """Instantiate or return the registered judge service for ``name``."""
        if name not in self._judges:
            raise SpecValidationError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message=f"Judge service {name} not found in registry.",
            )
        return self._instantiate(
            self._judges[name].factory, required_methods=("judge",)
        )

    def get_tool(self, name: str) -> ToolHandler:
        """Instantiate or return the registered runtime tool handler for ``name``."""
        registration = self.get_tool_registration(name)
        return self._instantiate(registration.factory, required_methods=())

    def iter_hook_registrations(self) -> list[HookRegistration]:
        """Return hook registrations ordered by priority then registration order."""
        return sorted(
            self._hooks,
            key=lambda registration: (
                registration.priority,
                registration.registration_order,
            ),
        )

    def iter_hooks(self) -> Iterator[PipelineHook]:
        """Yield hook instances in the same order used by pipeline execution."""
        for registration in self.iter_hook_registrations():
            yield registration.hook

    def _register_builtin_extractors(self) -> None:
        from themis.extractors.builtin import (
            BoxedTextExtractor,
            ChoiceLetterExtractor,
            FirstNumberExtractor,
            JsonSchemaExtractor,
            NormalizedTextExtractor,
            RegexExtractor,
        )

        self.register_extractor(
            "regex", RegexExtractor, version="1.0.0", plugin_api="1.0"
        )
        self.register_extractor(
            "json_schema", JsonSchemaExtractor, version="1.0.0", plugin_api="1.0"
        )
        self.register_extractor(
            "first_number", FirstNumberExtractor, version="1.0.0", plugin_api="1.0"
        )
        self.register_extractor(
            "choice_letter", ChoiceLetterExtractor, version="1.0.0", plugin_api="1.0"
        )
        self.register_extractor(
            "boxed_text", BoxedTextExtractor, version="1.0.0", plugin_api="1.0"
        )
        self.register_extractor(
            "normalized_text",
            NormalizedTextExtractor,
            version="1.0.0",
            plugin_api="1.0",
        )

    def _next_registration_order(self) -> int:
        self._registration_order += 1
        return self._registration_order

    def _instantiate(
        self,
        factory: Callable[[], _PluginT] | type[_PluginT] | _PluginT,
        *,
        required_methods: tuple[str, ...],
    ) -> _PluginT:
        if isinstance(factory, type):
            return cast(_PluginT, factory())
        if not callable(factory):
            return factory
        if all(hasattr(factory, method_name) for method_name in required_methods):
            return cast(_PluginT, factory)
        return cast(_PluginT, factory())

    def _validate_extractor_signature(
        self,
        name: str,
        extractor: Extractor,
    ) -> None:
        signature = inspect.signature(extractor.extract)
        parameters = tuple(signature.parameters.values())
        has_config_parameter = "config" in signature.parameters
        positional_count = sum(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        )
        has_varargs = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        if has_config_parameter or positional_count >= 3 or has_varargs:
            return
        raise SpecValidationError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=(
                f"Extractor '{name}' must accept (trial, candidate, config); "
                "legacy two-argument extractors are no longer supported."
            ),
        )
