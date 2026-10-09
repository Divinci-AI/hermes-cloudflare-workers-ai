"""Live model catalog for Workers AI.

``GET /accounts/{id}/ai/models/search`` is paginated and answers ``{"result": [...],
"result_info": {...}}`` - NOT the ``{"data": [...]}`` shape Hermes's default ``fetch_models``
expects, which is why this module exists. Do not "simplify" it away.

Each row carries two identifiers: an internal UUID ``id`` and the slug ``name`` (``@cf/...``).
Only the slug is usable as a model id, so slugs are recognised by prefix, never by length or shape.
"""

from __future__ import annotations

import math
from typing import Any, Callable

SEARCH_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
PER_PAGE = 50
MAX_PAGES = 20  # hard bound: a runaway catalog must not loop the picker

# Output task types that produce chat text. Matched exactly (case-insensitive) rather than by
# rejecting substrings: "Image-Text-to-Text" is a chat-capable multimodal model, while
# "Text-to-Image" is not. UNVERIFIED against the live catalog until M3.
CHAT_TASKS = frozenset({"text generation", "image-text-to-text"})

_SLUG_PREFIXES = ("@cf/", "@hf/")

FetchJson = Callable[[str], dict]


def search_url(account_id: str, page: int = 1, per_page: int = PER_PAGE) -> str:
    return f"{SEARCH_URL.format(account_id=account_id)}?per_page={per_page}&page={page}"


def is_slug(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_SLUG_PREFIXES)


def model_slug(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ("name", "id"):
        if is_slug(row.get(key)):
            return row[key]
    return None


def task_name(row: dict) -> str:
    task = row.get("task")
    if isinstance(task, dict):
        task = task.get("name")
    return task.strip().lower() if isinstance(task, str) else ""


def is_chat_model(row: Any) -> bool:
    return model_slug(row) is not None and task_name(row) in CHAT_TASKS


def parse_page(payload: Any) -> tuple[list[str], int | None]:
    """(chat model slugs on this page, total page count if the API said)."""
    if not isinstance(payload, dict):
        return [], None
    rows = payload.get("result")
    if isinstance(rows, dict):
        rows = rows.get("models") or rows.get("data")
    if not isinstance(rows, list):
        return [], None
    slugs = [model_slug(row) for row in rows if is_chat_model(row)]
    info = payload.get("result_info")
    total_pages = None
    if isinstance(info, dict):
        total, per_page = info.get("total_count"), info.get("per_page")
        if isinstance(total, int) and isinstance(per_page, int) and per_page > 0:
            total_pages = math.ceil(total / per_page)
    return [s for s in slugs if s], total_pages


def fetch_chat_models(fetch_json: FetchJson, account_id: str) -> list[str] | None:
    """All chat models for the account, in API order, de-duplicated; None if the first page failed."""
    seen: dict[str, None] = {}
    total_pages: int | None = None
    for page in range(1, MAX_PAGES + 1):
        try:
            payload = fetch_json(search_url(account_id, page))
        except Exception:
            if page == 1:
                return None
            break  # keep what we have rather than dropping the whole picker
        slugs, pages = parse_page(payload)
        total_pages = pages if pages is not None else total_pages
        for slug in slugs:
            seen.setdefault(slug)
        raw = payload.get("result") if isinstance(payload, dict) else None
        if not raw or (total_pages is not None and page >= total_pages):
            break
    return list(seen) or None
