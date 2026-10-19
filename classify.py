"""How Hermes should treat Cloudflare Workers AI failures it cannot classify correctly by itself.

Registered as ``ProviderProfile.classify_api_error``; Hermes consults it only for THIS provider's
errors, before its built-in pipeline, and falls through to that pipeline when we return ``None``.

One case needs it. Workers AI answers ``HTTP 429`` with code ``3040`` ("AiError: Capacity temporarily
exceeded, please try again") when the ACCOUNT is under contention. Hermes's generic 429 verdict is
``rate_limit`` with ``should_rotate_credential``, which is right for a per-key quota and wrong here:
the limit is shared by every key on the account, so rotating burns pooled credentials for nothing and
can leave the pool exhausted once the capacity clears. ``overloaded`` backs off and retries, and
leaves the credential alone.
"""

from __future__ import annotations

from typing import Any, Mapping

CAPACITY_CODE = "3040"
_CAPACITY_PHRASE = "capacity temporarily exceeded"


def _codes(body: Any) -> list[str]:
    """Error codes in a Cloudflare envelope (``errors: [{code}]``) or an OpenAI-style ``error: {code}``."""
    if not isinstance(body, dict):
        return []
    found = []
    errors = body.get("errors")
    if isinstance(errors, list):
        found += [str(e.get("code")) for e in errors if isinstance(e, dict) and e.get("code") is not None]
    inner = body.get("error")
    if isinstance(inner, dict) and inner.get("code") is not None:
        found.append(str(inner["code"]))
    return found


def is_capacity_error(*, status_code: Any, error_code: Any, message: Any, body: Any) -> bool:
    """True for the account-level capacity 429. Matches the structured code or the exact phrase, never a
    bare ``3040`` substring (a request id or a byte count could contain one)."""
    if status_code != 429:
        return False
    if str(error_code) == CAPACITY_CODE or CAPACITY_CODE in _codes(body):
        return True
    return _CAPACITY_PHRASE in f"{message or ''} {body if body is not None else ''}".lower()


def classify_api_error(
    error: Any, *, status_code: Any = None, error_code: Any = None, message: Any = None,
    body: Any = None, model: Any = None,
) -> Mapping[str, Any] | None:
    if is_capacity_error(status_code=status_code, error_code=error_code, message=message, body=body):
        return {"reason": "overloaded", "retryable": True, "should_rotate_credential": False}
    return None
