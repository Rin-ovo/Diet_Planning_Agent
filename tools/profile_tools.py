from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from storage.neo4j_bridge import format_profile_addon, graph_forbidden_for_check, merge_taboo_keywords
from storage.user_context import get_current_user_id

from ._util import as_list, dumps, err, load_profile, save_profile

LIST_KEYS = frozenset({"favorites", "taboos", "allergies", "preferred_cuisines"})
_STR_FIELDS = frozenset(
    {"goals", "notes", "current_mood", "mood_note", "sex", "activity_level"},
)


def tool_get_user_profile(params: dict[str, Any]) -> str:
    _ = params.get("user_id")
    return dumps(load_profile()) + format_profile_addon()


def _coerce_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def tool_update_user_profile(params: dict[str, Any]) -> str:
    patch = params.get("patch")
    if not isinstance(patch, dict):
        return err("缺少 patch 字典")
    list_op = params.get("list_op", "append")
    if list_op not in ("append", "set"):
        return err('list_op 须为 "append" 或 "set"')

    data = load_profile()
    now_iso = datetime.now(timezone.utc).isoformat()

    for k, v in patch.items():
        if k in LIST_KEYS:
            if list_op == "set":
                data[k] = [str(x) for x in as_list(v, k)]
            else:
                cur = list(data.get(k) or [])
                for x in as_list(v, k):
                    s = str(x)
                    if s not in cur:
                        cur.append(s)
                data[k] = cur
        elif k == "comfort_snacks_ok":
            data[k] = bool(v)
        elif k in _STR_FIELDS:
            data[k] = str(v) if v is not None else ""
        elif k == "weight_kg":
            f = _coerce_float(v)
            if f is not None:
                data["weight_kg"] = f
                data["weight_updated_at"] = now_iso
        elif k == "height_cm":
            f = _coerce_float(v)
            if f is not None:
                data["height_cm"] = f
        elif k == "age":
            n = _coerce_int(v)
            if n is not None:
                data["age"] = n
        elif k == "weight_updated_at":
            data[k] = str(v) if v is not None else ""
        else:
            data[k] = v
    save_profile(data)
    if "taboos" in patch:
        merge_taboo_keywords(get_current_user_id(), data.get("taboos") or [])
    return "已更新用户饮食档案。\n" + dumps(data)


def _violates(item: str, forbidden: list[str]) -> list[str]:
    item_l = item.strip().lower()
    hit: list[str] = []
    for f in forbidden:
        t = str(f).strip()
        if not t:
            continue
        tl = t.lower()
        if tl in item_l or item_l in tl:
            hit.append(t)
    return hit


def tool_check_taboos(params: dict[str, Any]) -> str:
    items = params.get("items")
    if not isinstance(items, list) or not items:
        return err("items 须为非空列表")
    extra = params.get("extra_forbidden") or []
    if not isinstance(extra, list):
        return err("extra_forbidden 须为列表")

    prof = load_profile()
    forbidden = [str(x) for x in (prof.get("taboos") or []) + (prof.get("allergies") or [])]
    forbidden.extend(str(x) for x in extra)
    for g in graph_forbidden_for_check():
        if g and g not in forbidden:
            forbidden.append(g)

    lines: list[str] = []
    for it in items:
        name = str(it).strip()
        if not name:
            continue
        bad = _violates(name, forbidden)
        if bad:
            lines.append(f"禁止推荐: 「{name}」触犯禁忌/过敏: {', '.join(bad)}")
        else:
            lines.append(f"可推荐: 「{name}」")
    return "\n".join(lines) if lines else "无候选项。"


def tool_log_mood(params: dict[str, Any]) -> str:
    mood = params.get("mood")
    if mood is None or str(mood).strip() == "":
        return err("缺少 mood")
    note = params.get("mood_note", "")
    data = load_profile()
    data["current_mood"] = str(mood).strip()
    data["mood_note"] = str(note).strip() if note is not None else ""
    save_profile(data)
    return f"已记录心情: {data['current_mood']}。备注: {data['mood_note'] or '无'}"
