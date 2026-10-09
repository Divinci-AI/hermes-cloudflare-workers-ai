"""The OpenAI-compatible client for Workers AI.

``ProviderProfile.create_client()`` is the one per-call hook both the main agent and the auxiliary
client consult, which makes it the only place the real, profile-specific URL can be applied (see
``endpoint.py``). Returning a client from it SKIPS core's own client setup, so what core would have
done is repeated here: keepalive HTTP client, and ``max_retries=0`` because the conversation loop owns
retries and honours Retry-After (SDK retries on top of it would double-retry and defeat long backoffs
for Cloudflare capacity errors).
"""

from __future__ import annotations

import sys
from typing import Any

from .endpoint import PLACEHOLDER_BASE_URL, WorkersAIConfigError, resolve_base_url


def _process_bootstrap() -> Any | None:
    """Hermes's client helpers, but only in a process the Hermes launcher started.

    ``agent.process_bootstrap`` imports ``hermes_bootstrap``, whose import-time launch preparation can
    rebuild the checkout's UI assets and sync dependencies over the network. That must never be
    triggered by a plugin: under the launcher ``hermes_bootstrap`` is already loaded, so require it.
    """
    if sys.modules.get("hermes_bootstrap") is None:
        return None
    try:
        from agent import process_bootstrap

        return process_bootstrap
    except Exception:
        return None


def _openai_class() -> Any:
    bootstrap = _process_bootstrap()
    if bootstrap is not None:
        return bootstrap.OpenAI  # lazy SDK proxy core (and its tests) patch
    from openai import OpenAI

    return OpenAI


def _keepalive_http_client(base_url: str) -> Any | None:
    bootstrap = _process_bootstrap()
    if bootstrap is None:
        return None
    try:
        return bootstrap.build_keepalive_http_client(base_url)
    except Exception:
        return None


def _unconfigured_http_client(message: str) -> Any:
    """An HTTP client that never touches the network and answers 400 with *message*.

    Surfaces a missing account id as the provider's own error text in the conversation, instead of a
    404 from a placeholder URL or a connection error that hides the cause.
    """
    import httpx

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": message, "type": "invalid_request_error"}}
        )

    return httpx.Client(transport=httpx.MockTransport(refuse))


def build_client(client_kwargs: dict[str, Any]) -> Any:
    kwargs = dict(client_kwargs)
    try:
        kwargs["base_url"] = resolve_base_url(str(kwargs.get("base_url") or ""))
    except WorkersAIConfigError as exc:
        kwargs["base_url"] = PLACEHOLDER_BASE_URL
        kwargs["http_client"] = _unconfigured_http_client(str(exc))
    else:
        if "http_client" not in kwargs:
            http_client = _keepalive_http_client(kwargs["base_url"])
            if http_client is not None:
                kwargs["http_client"] = http_client
    kwargs.setdefault("max_retries", 0)
    return _openai_class()(**kwargs)
