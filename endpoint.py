"""Where a Workers AI request goes, decided per call from the bound profile's env.

Workers AI's OpenAI-compatible surface is account-scoped::

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1

Hermes keeps ONE process-wide registry row per provider name, so a URL baked into the profile at
import time would be shared by every profile the process serves (the last one to load wins). The
profile therefore registers ``PLACEHOLDER_BASE_URL`` - identical in every home, and under
``api.cloudflare.com`` so a token can never leave Cloudflare - and the real URL is resolved here,
at the last hop, from whichever Hermes home is bound to the current call.
"""

from __future__ import annotations

import re

ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
BASE_URL_ENV = "CLOUDFLARE_BASE_URL"

# Registered with Hermes; never used for an actual request. Deliberately carries no account id.
PLACEHOLDER_BASE_URL = "https://api.cloudflare.com/client/v4/ai/v1"

_SAFE_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class WorkersAIConfigError(RuntimeError):
    """The bound Hermes profile is missing, or has a malformed, Workers AI setting."""


def read_env(name: str) -> str:
    """Value of *name* for the CURRENT Hermes home (``.env`` first, then scoped ``os.environ``)."""
    from hermes_cli.config import get_env_value_prefer_dotenv

    return (get_env_value_prefer_dotenv(name) or "").strip().strip("\"'")


def account_url(account_id: str) -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def bound_account_id() -> str:
    """The account id of the bound profile, or "" when unset. Raises if set but malformed."""
    account = read_env(ACCOUNT_ID_ENV)
    if account and not _SAFE_ACCOUNT_ID.match(account):
        raise WorkersAIConfigError(
            f"{ACCOUNT_ID_ENV} contains unexpected characters; expected the 32-character "
            "account id from the Cloudflare dashboard."
        )
    return account


def explicit_base_url(candidate: str = "") -> str:
    """A base URL the user chose deliberately, or "".

    That is ``candidate`` (core hands in ``model.base_url`` or the registry row) unless it is
    just our placeholder, else the profile's own ``CLOUDFLARE_BASE_URL`` proxy override.
    """
    candidate = (candidate or "").strip().rstrip("/")
    if candidate and candidate != PLACEHOLDER_BASE_URL:
        return candidate
    return read_env(BASE_URL_ENV).rstrip("/")


def resolve_base_url(candidate: str = "") -> str:
    """The inference base URL for the bound profile. Raises ``WorkersAIConfigError`` if unusable."""
    explicit = explicit_base_url(candidate)
    if explicit:
        return explicit
    account = bound_account_id()
    if not account:
        raise WorkersAIConfigError(
            f"{ACCOUNT_ID_ENV} is not set for this Hermes profile. Add it to this profile's .env "
            "(Cloudflare dashboard -> Workers AI -> account id), or set "
            f"{BASE_URL_ENV} to use a proxy."
        )
    return account_url(account)
