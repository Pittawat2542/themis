"""Shared prompt rendering helpers for the benchmark-first API."""

from __future__ import annotations

from collections.abc import Mapping

from themis.specs.experiment import PromptMessage


class _ValueProxy:
    def __init__(self, value: object, *, placeholder: str | None = None) -> None:
        self._value = value
        self._placeholder = placeholder

    def __getattr__(self, name: str) -> "_ValueProxy":
        if isinstance(self._value, Mapping) and name in self._value:
            return _ValueProxy(self._value[name])
        if self._placeholder is not None:
            return _ValueProxy(self._placeholder)
        raise AttributeError(name)

    def __getitem__(self, key: str) -> "_ValueProxy":
        if isinstance(self._value, Mapping) and key in self._value:
            return _ValueProxy(self._value[key])
        if self._placeholder is not None:
            return _ValueProxy(self._placeholder)
        raise KeyError(key)

    def __str__(self) -> str:
        if self._placeholder is not None and self._value == self._placeholder:
            return self._placeholder
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        if self._placeholder is not None and self._value == self._placeholder:
            return self._placeholder
        return format(self._value, format_spec)


class _SafeNamespace(dict[str, object]):
    def __missing__(self, key: str) -> _ValueProxy:
        return _ValueProxy("{" + key + "}", placeholder="{" + key + "}")


def render_prompt_messages(
    messages: list[PromptMessage],
    namespaces: Mapping[str, object],
    *,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Render prompt messages against benchmark-native namespaces."""

    context: dict[str, object] = {}
    for key, value in namespaces.items():
        context[key] = _ValueProxy(value)
    mapping = _SafeNamespace(context)
    rendered: list[dict[str, str]] = []
    for message in messages:
        if strict:
            content = message.content.format_map(context)
        else:
            content = message.content.format_map(mapping)
        rendered.append({"role": message.role.value, "content": content})
    return rendered
