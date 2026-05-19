"""
Neo4j 与饮食 agent 的桥接：默认只读（查忌口、健康检查），不向图写入、不自动建约束/索引。

环境变量：
  NEO4J_READ_ONLY   默认 1（true）：不向 Neo4j MERGE、不调用 ensure_user_taboo_schema。
                    设为 0 / false / no / off 时，允许 merge_taboo_keywords 写入（仅建议本地/自建库）。

失败时静默降级（不阻塞 JSON 档案流程）。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable

from .neo4j_store import Neo4jStore
from .user_context import get_current_user_id

logger = logging.getLogger(__name__)

_cached_store: Neo4jStore | None | bool = False  # False = 尚未解析


def is_neo4j_agent_read_only() -> bool:
    """Agent 集成是否禁止写 Neo4j（默认禁止）。"""
    v = os.getenv("NEO4J_READ_ONLY", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _active_store() -> Neo4jStore | None:
    global _cached_store
    if _cached_store is False:
        s = Neo4jStore.from_env()
        _cached_store = s if s.enabled else None
    return _cached_store if isinstance(_cached_store, Neo4jStore) else None


def neo4j_health() -> dict[str, Any]:
    """连接与连通性；附带 agent 只读策略说明。"""
    h = Neo4jStore.from_env().health()
    ro = is_neo4j_agent_read_only()
    h["agent_read_only"] = ro
    if ro:
        h["agent_writes"] = False
        h["agent_policy"] = "只读：仅 execute_read / list_user_taboos；不写图、不建约束"
    else:
        h["agent_writes"] = True
        h["agent_policy"] = "可写：update_user_profile(taboos) 等会 MERGE 忌口边（不设 NEO4J_READ_ONLY=1 时）"
    return h


def merge_taboo_keywords(user_id: str, keywords: Iterable[str]) -> dict[str, Any]:
    """
    将忌口词写入图（仅当 NEO4J_READ_ONLY=0）。
    默认只读模式下立即返回，不修改 Neo4j。
    """
    if is_neo4j_agent_read_only():
        return {"ok": False, "reason": "NEO4J_READ_ONLY=1（默认）：Agent 不向 Neo4j 写入", "merged": 0}
    store = _active_store()
    if store is None:
        return {"ok": False, "reason": "neo4j 未启用（缺少 neo4j 包或未设置 NEO4J_PASSWORD）"}
    uid = str(user_id).strip()
    if not uid:
        return {"ok": False, "reason": "empty user_id"}
    merged = 0
    errors: list[str] = []
    for raw in keywords:
        kw = str(raw).strip()
        if not kw:
            continue
        try:
            store.merge_user_avoids_keyword(uid, kw)
            merged += 1
        except Exception as e:
            errors.append(f"{kw!r}: {e}")
    return {"ok": not errors, "merged": merged, "errors": errors[:5]}


def list_graph_taboo_keywords(user_id: str) -> list[str]:
    store = _active_store()
    if store is None:
        return []
    try:
        return store.list_user_taboos(str(user_id).strip())
    except Exception as e:
        logger.debug("Neo4j list_user_taboos: %s", e)
        return []


def format_profile_addon(user_id: str | None = None) -> str:
    """附在 get_user_profile 文本后的 Neo4j 状态与图中忌口列表。"""
    uid = (user_id or get_current_user_id()).strip()
    h = neo4j_health()
    lines = ["\n---", "【Neo4j 图存储】", json.dumps(h, ensure_ascii=False)]
    if not h.get("enabled"):
        return "\n".join(lines)
    kws = list_graph_taboo_keywords(uid)
    if kws:
        cap = 24
        head = kws[:cap]
        tail = f"…（共 {len(kws)} 条）" if len(kws) > cap else ""
        lines.append("图中忌口关键词: " + ", ".join(head) + tail)
    else:
        lines.append("图中尚无该用户的 AVOIDS 忌口边。")
    if is_neo4j_agent_read_only():
        lines.append("说明: Agent 对 Neo4j 为只读，不会 MERGE/建约束；忌口写入请用本地 profile 或 DBA 维护图。")
    return "\n".join(lines)


def graph_forbidden_for_check(user_id: str | None = None) -> list[str]:
    """供 check_taboos 合并进 forbidden 列表（与 JSON taboos 并列）。"""
    uid = (user_id or get_current_user_id()).strip()
    return list_graph_taboo_keywords(uid)
