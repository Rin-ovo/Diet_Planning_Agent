"""当前会话所服务的饮食用户（画像/推荐/智能记忆均按此隔离）。"""
from __future__ import annotations

import contextvars
import re

_CURRENT_USER_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "diet_user_id",
    default="me",
)

# 对外 slug：我 -> me，张三 -> zhangsan，李四 -> lisi
_USER_ALIASES: dict[str, str] = {
    "me": "me",
    "我": "me",
    "default": "me",
    "zhangsan": "zhangsan",
    "张三": "zhangsan",
    "lisi": "lisi",
    "李四": "lisi",
}

DISPLAY_NAME: dict[str, str] = {
    "me": "我",
    "zhangsan": "张三",
    "lisi": "李四",
}

ALLOWED_USER_IDS: frozenset[str] = frozenset({"me", "zhangsan", "lisi"})


def normalize_user_id(raw: str) -> str:
    s = str(raw).strip().lower()
    if not s:
        return "me"
    if s in _USER_ALIASES:
        return _USER_ALIASES[s]
    # 允许直接传中文昵称
    for k, v in _USER_ALIASES.items():
        if k.lower() == s:
            return v
    return s


def is_allowed_user_id(uid: str) -> bool:
    return normalize_user_id(uid) in ALLOWED_USER_IDS


def set_current_user_id(user_id: str) -> str:
    uid = normalize_user_id(user_id)
    if uid not in ALLOWED_USER_IDS:
        raise ValueError(f"不支持的用户 id: {user_id!r}，可选: me(我), zhangsan(张三), lisi(李四)")
    _CURRENT_USER_ID.set(uid)
    return uid


def get_current_user_id() -> str:
    return _CURRENT_USER_ID.get()


def get_current_user_display() -> str:
    return DISPLAY_NAME.get(get_current_user_id(), get_current_user_id())
