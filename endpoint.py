"""Where a Workers AI request goes, decided per call from the bound profile's env.

Workers AI's OpenAI-compatible surface is account-scoped::

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1

Hermes keeps ONE process-wide registry row per provider name, so a URL baked into the profile at
import time would be shared by every profile the process serves (the last one to load wins). The
profile therefore registers ``PLACEHOLDER_BASE_URL`` - identical in every home, and under
``api.cloudflare.com`` - and the real URL is resolved here, at the last hop, from whichever
Hermes home is bound to the current call.

The bearer token goes wherever the resolved URL points, so a URL that does NOT point at Cloudflare is
only accepted when the user chose it deliberately, and only in a form that cannot leak the token in
transit (see ``validate_base_url``): ``CLOUDFLARE_BASE_URL`` is read from the bound profile's ``.env``
and never from the process environment, because a variable exported in a shell (or inherited from a
parent process) would otherwise silently redirect the token of every profile.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
BASE_URL_ENV = "CLOUDFLARE_BASE_URL"

# Registered with Hermes; never used for an actual request. Deliberately carries no account id.
PLACEHOLDER_BASE_URL = "https://api.cloudflare.com/client/v4/ai/v1"

_SAFE_ACCOUNT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Plain http is accepted only for a service on this machine (a local proxy / gateway under test).
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class WorkersAIConfigError(RuntimeError):
    """The bound Hermes profile is missing, or has a malformed, Workers AI setting."""


def _clean(value: str | None) -> str:
    return (value or "").strip().strip("\"'")


def read_env(name: str) -> str:
    """Value of *name* for the CURRENT Hermes home (``.env`` first, then scoped ``os.environ``)."""
    from hermes_cli.config import get_env_value_prefer_dotenv

    return _clean(get_env_value_prefer_dotenv(name))


def read_profile_env(name: str) -> str:
    """Value of *name* from the bound profile's ``.env`` ONLY - never the process environment.

    For settings that redirect where the token is sent. An unresolved ``op://`` reference is not a
    URL and fails validation downstream rather than being replaced by an ambient value.
    """
    from hermes_cli.config import load_env

    return _clean(load_env().get(name))


def validate_base_url(url: str, *, source: str) -> str:
    """*url* without a trailing slash, or ``WorkersAIConfigError`` if it could leak the token.

    Requires ``https`` (``http`` only for localhost / 127.0.0.1 / ::1), a host, and no embedded
    credentials, query or fragment. The error names *source* and never the URL itself: a rejected
    value may carry a password, and this message lands in the conversation and in logs.
    """
    problem = None
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError:
        problem = "is not a valid URL"
    else:
        if any(ch.isspace() or ord(ch) < 32 for ch in url):
            problem = "contains whitespace or control characters"
        elif not host:
            problem = "has no host"
        elif parts.scheme == "http" and host not in _LOOPBACK_HOSTS:
            problem = "uses plain http, which would send the API token unencrypted"
        elif parts.scheme not in ("https", "http"):
            problem = "must start with https://"
        elif parts.username is not None or parts.password is not None:
            problem = "contains embedded credentials"
        elif parts.query or parts.fragment:
            problem = "contains a query string or fragment"
    if problem:
        raise WorkersAIConfigError(
            f"{source} {problem}. Use an https:// base URL without credentials "
            "(http:// is allowed only for localhost)."
        )
    return url.rstrip("/")


def account_url(account_id: str) -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def bound_account_id() -> str:
    """The account id of the bound profile, or "" when unset. Raises if set but malformed."""
    account = read_env(ACCOUNT_ID_ENV)
    if account and not _SAFE_ACCOUNT_ID.fullmatch(account):
        raise WorkersAIConfigError(
            f"{ACCOUNT_ID_ENV} contains unexpected characters; expected the 32-character "
            "account id from the Cloudflare dashboard."
        )
    return account


def explicit_base_url(candidate: str = "") -> str:
    """A validated base URL the user chose deliberately, or "".

    That is ``candidate`` (core hands in ``model.base_url``, a pooled credential's URL or the
    registry row) unless it is just our placeholder, else the profile's own ``CLOUDFLARE_BASE_URL``
    proxy override from its ``.env``. Raises ``WorkersAIConfigError`` if the chosen URL is unsafe.
    """
    candidate = (candidate or "").strip().rstrip("/")
    if candidate and candidate != PLACEHOLDER_BASE_URL:
        return validate_base_url(candidate, source="The configured base URL")
    override = read_profile_env(BASE_URL_ENV)
    if not override:
        return ""
    return validate_base_url(override, source=BASE_URL_ENV)


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
