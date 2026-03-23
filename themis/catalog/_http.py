"""Shared stdlib HTTP helpers for catalog transports."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast
from urllib import request


DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0


type CatalogUrlopen = Callable[..., object]


def load_json_url(
    url: str,
    *,
    urlopen: CatalogUrlopen = request.urlopen,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> object:
    with _urlopen_with_optional_timeout(
        urlopen,
        request.Request(url),
        timeout=timeout,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_jsonl_url(
    url: str,
    *,
    urlopen: CatalogUrlopen = request.urlopen,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
):
    with _urlopen_with_optional_timeout(
        urlopen,
        request.Request(url),
        timeout=timeout,
    ) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            yield cast(object, json.loads(raw_line))


def open_json_request(
    req: request.Request,
    *,
    urlopen: CatalogUrlopen = request.urlopen,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> object:
    with _urlopen_with_optional_timeout(urlopen, req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _urlopen_with_optional_timeout(
    urlopen: CatalogUrlopen,
    req: request.Request,
    *,
    timeout: float,
):
    try:
        return urlopen(req, timeout=timeout)
    except TypeError:
        return urlopen(req)
