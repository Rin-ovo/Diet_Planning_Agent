from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ._util import dumps, err, load_profile, save_profile

_RECOMMENDATION_CAP = 800
_PRUNE_OLDER_THAN_DAYS = 60


def _parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _normalize(s: str) -> str:
    return str(s).strip().lower()


def _within_cutoff(entry_at: str, cutoff: datetime) -> bool:
    dt = _parse_dt(entry_at)
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff


def _recent_item_names(within_days: int) -> set[str]:
    prof = load_profile()
    hist = prof.get("recommendation_history") or []
    if not isinstance(hist, list):
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=within_days)
    names: set[str] = set()
    for e in hist:
        if not isinstance(e, dict):
            continue
        at = e.get("at")
        if not at or not _within_cutoff(str(at), cutoff):
            continue
        item = _normalize(str(e.get("item", "")))
        if item:
            names.add(item)
    return names


def _candidate_blocked(name: str, recent: set[str]) -> bool:
    key = _normalize(name)
    if not key:
        return False
    if key in recent:
        return True
    for r in recent:
        if not r:
            continue
        if len(r) >= 2 and (r in key or key in r):
            return True
    return False


def tool_get_recent_recommendations(params: dict[str, Any]) -> str:
    days = int(params.get("within_days", 3) or 3)
    if days < 1 or days > 30:
        return err("within_days 建议在 1~30")
    prof = load_profile()
    hist = prof.get("recommendation_history") or []
    if not isinstance(hist, list):
        hist = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent: list[dict[str, Any]] = []
    for e in hist:
        if not isinstance(e, dict):
            continue
        if not e.get("at") or not _within_cutoff(str(e["at"]), cutoff):
            continue
        recent.append({"item": e.get("item", ""), "at": e.get("at", "")})
    return dumps({"within_days": days, "recent_recommendations": recent})


def tool_filter_recent_recommendations(params: dict[str, Any]) -> str:
    candidates = params.get("candidates")
    if not isinstance(candidates, list):
        return err("candidates 须为列表")
    days = int(params.get("within_days", 3) or 3)
    if days < 1 or days > 30:
        return err("within_days 建议在 1~30")
    recent = _recent_item_names(days)
    ok: list[str] = []
    blocked: list[str] = []
    for c in candidates:
        s = str(c).strip()
        if not s:
            continue
        if _candidate_blocked(s, recent):
            blocked.append(s)
        else:
            ok.append(s)
    return dumps(
        {
            "within_days": days,
            "可推荐": ok,
            "近N天已推荐建议避开": blocked,
        }
    )


def tool_record_recommended_items(params: dict[str, Any]) -> str:
    items = params.get("items")
    if not isinstance(items, list) or not items:
        return err("items 须为非空列表")
    prof = load_profile()
    hist: list[dict[str, Any]] = list(prof.get("recommendation_history") or [])
    if not isinstance(hist, list):
        hist = []
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for it in items:
        s = str(it).strip()
        if not s:
            continue
        hist.append({"item": s, "at": now})
        added += 1
    cutoff_prune = datetime.now(timezone.utc) - timedelta(days=_PRUNE_OLDER_THAN_DAYS)
    pruned: list[dict[str, Any]] = []
    for e in hist:
        if not isinstance(e, dict):
            continue
        at = e.get("at")
        if at and _within_cutoff(str(at), cutoff_prune):
            pruned.append(e)
    prof["recommendation_history"] = pruned[-_RECOMMENDATION_CAP:]
    save_profile(prof)
    return f"已记录 {added} 条推荐（UTC 时间）。当前保留约 {len(prof['recommendation_history'])} 条历史条目。"
