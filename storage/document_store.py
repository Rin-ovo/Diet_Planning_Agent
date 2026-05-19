"""
按用户隔离的文档存储：画像 profile.json、推荐流水 recommendation_history.json。
后续可将向量/图检索接在 neo4j_store 等模块，由本模块统一编排入口。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .paths import project_data_dir
from .user_context import ALLOWED_USER_IDS, DISPLAY_NAME, normalize_user_id


def users_root() -> Path:
    p = project_data_dir() / "users"
    p.mkdir(parents=True, exist_ok=True)
    return p


class DocumentStore:
    def __init__(self, user_id: str):
        self.user_id = normalize_user_id(user_id)
        if self.user_id not in ALLOWED_USER_IDS:
            raise ValueError(f"未知用户: {user_id!r}")

    @property
    def user_dir(self) -> Path:
        p = users_root() / self.user_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def profile_path(self) -> Path:
        return self.user_dir / "profile.json"

    def recommendation_history_path(self) -> Path:
        return self.user_dir / "recommendation_history.json"

    def smart_memory_path(self) -> Path:
        return self.user_dir / "smart_memory.json"


def legacy_profile_path() -> Path:
    return project_data_dir() / "diet_profile.json"


def legacy_global_recommendation_path() -> Path:
    return project_data_dir() / "recommendation_history.json"


def legacy_global_smart_memory_path() -> Path:
    return project_data_dir() / "smart_memory.json"


def migrate_legacy_to_me_user() -> bool:
    """
    若存在旧版 data/diet_profile.json 且尚无 data/users/me/profile.json，
    则迁移画像、推荐流水、根目录 smart_memory 到 me 用户目录。返回是否执行了迁移。
    """
    me_profile = DocumentStore("me").profile_path()
    if me_profile.is_file():
        return False
    leg = legacy_profile_path()
    if not leg.is_file():
        return False

    store_me = DocumentStore("me")
    store_me.user_dir.mkdir(parents=True, exist_ok=True)

    with leg.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raw = {}

    hist_inline = raw.pop("recommendation_history", None)
    body = dict(raw)
    if "profile_display_name" not in body or not str(body.get("profile_display_name", "")).strip():
        body["profile_display_name"] = DISPLAY_NAME["me"]

    with store_me.profile_path().open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)

    # 推荐：优先根目录侧车，否则内嵌
    entries: list[dict[str, Any]] = []
    g = legacy_global_recommendation_path()
    if g.is_file():
        try:
            with g.open("r", encoding="utf-8") as f:
                pack = json.load(f)
            if isinstance(pack, dict):
                ent = pack.get("entries")
                if isinstance(ent, list):
                    entries = [e for e in ent if isinstance(e, dict)]
        except (OSError, json.JSONDecodeError):
            entries = []
    elif isinstance(hist_inline, list):
        entries = [e for e in hist_inline if isinstance(e, dict)]

    payload = {"version": 1, "entries": entries}
    with store_me.recommendation_history_path().open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    sm = legacy_global_smart_memory_path()
    if sm.is_file():
        shutil.copy2(sm, store_me.smart_memory_path())

    return True


def default_profile_dict() -> dict[str, Any]:
    """与 tools._util.default_profile 字段一致（避免 tools 导入 storage 时环依赖）。"""
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


def ensure_preset_users_exist() -> None:
    """创建 me / zhangsan / lisi 的空白画像（若不存在）。"""
    presets = (
        ("me", "我"),
        ("zhangsan", "张三"),
        ("lisi", "李四"),
    )
    for uid, display in presets:
        store = DocumentStore(uid)
        if store.profile_path().is_file():
            continue
        prof = default_profile_dict()
        prof["profile_display_name"] = display
        prof["recommendation_history"] = []
        body = {k: v for k, v in prof.items() if k != "recommendation_history"}
        with store.profile_path().open("w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
        with store.recommendation_history_path().open("w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": []}, f, ensure_ascii=False, indent=2)
