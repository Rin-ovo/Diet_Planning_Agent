from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storage.document_store import (
    DocumentStore,
    ensure_preset_users_exist,
    migrate_legacy_to_me_user,
)
from storage.paths import project_data_dir
from storage.user_context import get_current_user_id


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def profile_path() -> Path:
    return DocumentStore(get_current_user_id()).profile_path()


def recommendation_history_path() -> Path:
    return DocumentStore(get_current_user_id()).recommendation_history_path()


def _read_recommendation_history_entries() -> list[dict[str, Any]]:
    p = recommendation_history_path()
    if not p.is_file():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            ent = raw.get("entries")
            if isinstance(ent, list):
                return [e for e in ent if isinstance(e, dict)]
        if isinstance(raw, list):
            return [e for e in raw if isinstance(e, dict)]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _write_recommendation_history_entries(entries: list[dict[str, Any]]) -> None:
    p = recommendation_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "entries": entries}
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def profile_seed_jsonl_path() -> Path:
    return project_data_dir() / "profile_seed.jsonl"


def _merge_lines_from_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    merged: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                merged.update(obj)
    return merged


def default_profile() -> dict[str, Any]:
    return {
        "profile_display_name": "",
        "favorites": [],
        "taboos": [],
        "allergies": [],
        "goals": "",
        "preferred_cuisines": [],
        "notes": "",
        "comfort_snacks_ok": True,
        "current_mood": "",
        "mood_note": "",
        "recommendation_history": [],
        "weight_kg": None,
        "height_cm": None,
        "age": None,
        "sex": "",
        "activity_level": "",
        "weight_updated_at": "",
        "weekly_meal_plan": None,
    }


def load_profile() -> dict[str, Any]:
    migrate_legacy_to_me_user()
    ensure_preset_users_exist()

    base = default_profile()
    p = profile_path()
    seed = _merge_lines_from_jsonl(profile_seed_jsonl_path())

    if not p.is_file():
        base.update(seed)
        if not str(base.get("profile_display_name", "")).strip():
            from storage.user_context import DISPLAY_NAME

            uid = get_current_user_id()
            base["profile_display_name"] = DISPLAY_NAME.get(uid, uid)
        save_profile(base)
        return dict(base)

    with p.open("r", encoding="utf-8") as f:
        raw_obj = json.load(f)
    raw: dict[str, Any] = raw_obj if isinstance(raw_obj, dict) else {}
    base.update(raw)
    for k, v in default_profile().items():
        if k not in base:
            base[k] = v

    hist, need_resync = _resolve_recommendation_history(base, raw)
    base["recommendation_history"] = hist
    if need_resync:
        save_profile(base)
    return base


def _resolve_recommendation_history(
    base: dict[str, Any],
    raw_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    side = recommendation_history_path()
    if side.is_file():
        entries = _read_recommendation_history_entries()
        stale_inline = "recommendation_history" in raw_profile
        return entries, stale_inline

    inner = base.get("recommendation_history")
    if isinstance(inner, list) and inner:
        _write_recommendation_history_entries(inner)
        return inner, True
    return inner if isinstance(inner, list) else [], False


def save_profile(data: dict[str, Any]) -> None:
    hist = data.get("recommendation_history")
    if not isinstance(hist, list):
        hist = []
    _write_recommendation_history_entries(hist)
    body = {k: v for k, v in data.items() if k != "recommendation_history"}
    profile_path().parent.mkdir(parents=True, exist_ok=True)
    with profile_path().open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)


def as_list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def err(msg: str) -> str:
    return f"[错误] {msg}"
