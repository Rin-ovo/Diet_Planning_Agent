from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from storage.neo4j_bridge import merge_taboo_keywords
from storage.user_context import get_current_user_id

from ._util import dumps, err, load_profile, save_profile


def tool_add_plan_feedback_exclusions(params: dict[str, Any]) -> str:
    items = params.get("dislike_items") or []
    kws = params.get("dislike_keywords") or []
    if not isinstance(items, list):
        return err("dislike_items 须为列表")
    if not isinstance(kws, list):
        return err("dislike_keywords 须为列表")
    to_add: list[str] = []
    for x in list(items) + list(kws):
        s = str(x).strip()
        if s and s not in to_add:
            to_add.append(s)
    if not to_add:
        return err("至少提供 dislike_items 或 dislike_keywords 之一")

    data = load_profile()
    tab = list(data.get("taboos") or [])
    for s in to_add:
        if s not in tab:
            tab.append(s)
    data["taboos"] = tab

    note = params.get("reason_note")
    if note and str(note).strip():
        old = str(data.get("notes") or "").strip()
        line = f"[计划反馈 {datetime.now(timezone.utc).strftime('%Y-%m-%d')}] {str(note).strip()}"
        data["notes"] = (old + "\n" + line).strip() if old else line

    save_profile(data)
    merge_taboo_keywords(get_current_user_id(), to_add)
    return "已写入禁忌（后续计划与推荐将避开）：\n" + dumps(to_add) + "\n当前 taboos 总数: " + str(len(tab))
