from __future__ import annotations

from typing import Any

from ._util import dumps, load_profile

COMFORT_CATALOG: list[dict[str, str]] = [
    {"name": "低糖奶茶（小杯）", "note": "注意额外糖分与热量"},
    {"name": "黑巧克力一小块", "note": "相对可控份量"},
    {"name": "水果酸奶", "note": "选无糖/低糖酸奶更佳"},
    {"name": "香蕉", "note": "补充碳水与钾"},
    {"name": "坚果一小把", "note": "高热量，控制份量"},
    {"name": "热牛奶", "note": "乳糖不耐请换植物奶"},
]


def _hits_taboo(name: str, forbidden: list[str]) -> bool:
    for f in forbidden:
        t = str(f).strip()
        if t and t in name:
            return True
    return False


def tool_suggest_comfort_foods(params: dict[str, Any]) -> str:
    prof = load_profile()
    if not prof.get("comfort_snacks_ok", True):
        return "档案中已关闭安慰型零食/奶茶推荐。"

    mood = params.get("mood", prof.get("current_mood", ""))
    mood_s = str(mood).strip().lower()
    force = bool(params.get("force", False))
    bad_signals = ("bad", "low", "差", "低落", "不好", "郁闷", "难过", "丧")
    is_bad = force or any(s in mood_s for s in bad_signals)

    if not is_bad:
        return (
            "未识别为心情不好（可 mood=低落 或 force=true）。档案心情: "
            + str(prof.get("current_mood") or "未记录")
        )

    forbidden = [str(x) for x in (prof.get("taboos") or []) + (prof.get("allergies") or [])]
    ok = [c for c in COMFORT_CATALOG if not _hits_taboo(c["name"], forbidden)]
    if not ok:
        return "候选均与禁忌冲突。"
    return "心情不好时参考:\n" + dumps(ok)
