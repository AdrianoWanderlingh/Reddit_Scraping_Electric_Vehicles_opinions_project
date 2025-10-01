from __future__ import annotations

import datetime as dt
from typing import Any, Dict


def _parse_created_utc(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compose_text(obj: Dict[str, Any], is_submission: bool) -> str:
    if is_submission:
        title = obj.get("title") or ""
        selftext = obj.get("selftext") or ""
        pieces = [title.strip(), selftext.strip()]
        return "\n\n".join([piece for piece in pieces if piece])
    body = obj.get("body")
    if isinstance(body, str):
        return body.strip()
    return ""


def _resolve_permalink(obj: Dict[str, Any], is_submission: bool) -> str:
    permalink = obj.get("permalink")
    if isinstance(permalink, str) and permalink:
        return permalink if permalink.startswith("/") else f"/{permalink.lstrip('/')}"
    subreddit = obj.get("subreddit") or "unknown"
    if is_submission:
        post_id = obj.get("id") or ""
        return f"/r/{subreddit}/comments/{post_id}/"
    link_id = obj.get("link_id")
    comment_id = obj.get("id")
    if isinstance(link_id, str) and link_id.startswith("t3_") and comment_id:
        return f"/r/{subreddit}/comments/{link_id[3:]}/_/{comment_id}/"
    return ""


def normalize(obj: Dict[str, Any], ideology_map: Dict[str, str]) -> Dict[str, Any]:
    utc = _parse_created_utc(obj.get("created_utc"))
    dt_utc = dt.datetime.utcfromtimestamp(utc) if utc else None

    subreddit = obj.get("subreddit") or ""
    ideology = ideology_map.get(subreddit.lower())

    is_submission = "selftext" in obj or "title" in obj

    text = _compose_text(obj, is_submission)

    record = {
        "id": str(obj.get("id")) if obj.get("id") is not None else None,
        "is_submission": bool(is_submission),
        "created_utc": utc,
        "year": dt_utc.year if dt_utc else None,
        "month": dt_utc.month if dt_utc else None,
        "subreddit": subreddit,
        "ideology_group": ideology,
        "author": obj.get("author"),
        "text": text,
        "permalink": _resolve_permalink(obj, is_submission),
        "score": _parse_int(obj.get("score")),
    }
    return record
