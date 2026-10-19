"""Cloudflare Workers AI as a Hermes model provider (standalone user plugin).

Setup: put ``CLOUDFLARE_API_TOKEN`` (or ``CLOUDFLARE_API_KEY``) and ``CLOUDFLARE_ACCOUNT_ID`` in the
Hermes profile's ``.env``. See README.md; design notes are in ``endpoint.py`` and ``client.py``.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

from .catalog import fetch_chat_models
from .client import build_client
from .endpoint import PLACEHOLDER_BASE_URL, WorkersAIConfigError, bound_account_id, explicit_base_url

NAME = "cloudflare-workers-ai"


class CloudflareWorkersAIProfile(ProviderProfile):
    """Workers AI. Registered with a placeholder URL; the real one is resolved per call."""

    def create_client(self, **client_kwargs: Any) -> Any:
        return build_client(client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        try:
            override = explicit_base_url(base_url or "")
        except WorkersAIConfigError:
            return None  # never send the key to a URL that failed validation
        if override:
            # A proxy is selected: its own /models is the catalog (base-class contract).
            return super().fetch_models(api_key=api_key, base_url=override, timeout=timeout)
        if not api_key:
            return None
        try:
            account = bound_account_id()
        except WorkersAIConfigError:
            return None
        if not account:
            return None
        return fetch_chat_models(_json_fetcher(api_key, timeout), account)


def _json_fetcher(api_key: str, timeout: float):
    def fetch(url: str) -> dict:
        from hermes_cli.urllib_security import open_credentialed_url

        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", _profile_user_agent())
        with open_credentialed_url(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    return fetch


register_provider(
    CloudflareWorkersAIProfile(
        name=NAME,
        aliases=("cf-workers-ai",),
        display_name="Cloudflare Workers AI",
        description="Cloudflare Workers AI (OpenAI-compatible, @cf/... models)",
        signup_url="https://developers.cloudflare.com/workers-ai/get-started/rest-api/",
        # Order matters: first non-empty wins. No *_BASE_URL here on purpose - core reads that from
        # the PROCESS env, which is how one profile's proxy would catch another profile's token.
        env_vars=("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY"),
        base_url=PLACEHOLDER_BASE_URL,
        supports_health_check=False,  # core would probe the placeholder URL; see README
        # One id known to work on a live account; the full list is verified against the live catalog
        # before release (plan M3) rather than copied from elsewhere.
        fallback_models=("@cf/deepseek-ai/deepseek-v4-flash-0731",),
    )
)
