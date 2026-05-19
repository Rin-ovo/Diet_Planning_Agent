"""
Neo4j 图存储：通用 Cypher 读写 + 饮食域便捷方法（用户—忌口食材）。
依赖：pip install neo4j
环境变量：
  NEO4J_URI       默认 neo4j://localhost:7687
  NEO4J_USER      默认 neo4j
  NEO4J_PASSWORD  必填（未设置则 health 为不可用）

Agent 侧只读策略见 neo4j_bridge（默认不向图写入、不自动建约束）。
execute_read 使用驱动 READ 会话，避免误连到写路由。
"""
from __future__ import annotations

import os
from typing import Any

try:
    from neo4j import READ_ACCESS, GraphDatabase
    from neo4j.exceptions import Neo4jError

    _HAS_NEO4J = True
except ImportError:
    READ_ACCESS = None  # type: ignore[misc, assignment]
    GraphDatabase = None  # type: ignore[misc, assignment]
    Neo4jError = Exception  # type: ignore[misc, assignment]
    _HAS_NEO4J = False


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {k: record[k] for k in record.keys()}


class Neo4jStore:
    """封装 Neo4j Driver：参数化查询 + 简单图模式写入。"""

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = (uri or os.getenv("NEO4J_URI") or "neo4j://localhost:7687").strip()
        self._user = (user or os.getenv("NEO4J_USER") or "neo4j").strip()
        self._password = (password or os.getenv("NEO4J_PASSWORD") or "").strip()
        self._driver: Any = None

    @classmethod
    def from_env(cls) -> Neo4jStore:
        return cls()

    def _require_package(self) -> None:
        if not _HAS_NEO4J:
            raise RuntimeError("未安装 neo4j 驱动，请执行: pip install neo4j")

    def _get_driver(self) -> Any:
        self._require_package()
        if not self._password:
            raise RuntimeError("未设置 NEO4J_PASSWORD，无法连接 Neo4j")
        if self._driver is None:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    @property
    def enabled(self) -> bool:
        return _HAS_NEO4J and bool(self._password)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health(self) -> dict[str, Any]:
        if not _HAS_NEO4J:
            return {"backend": "neo4j", "enabled": False, "reason": "缺少依赖 neo4j"}
        if not self._password:
            return {"backend": "neo4j", "enabled": False, "reason": "未设置 NEO4J_PASSWORD"}
        try:
            drv = self._get_driver()
            drv.verify_connectivity()
            return {"backend": "neo4j", "enabled": True, "uri": self._uri, "user": self._user}
        except Exception as e:
            return {"backend": "neo4j", "enabled": False, "uri": self._uri, "error": str(e)}

    def execute_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """只读查询，返回多行 dict（READ 路由，禁止依赖写库）。"""
        drv = self._get_driver()
        p = params or {}
        mode = READ_ACCESS if _HAS_NEO4J and READ_ACCESS is not None else None
        kwargs: dict[str, Any] = {}
        if mode is not None:
            kwargs["default_access_mode"] = mode
        with drv.session(**kwargs) as session:
            result = session.run(cypher, p)
            return [_record_to_dict(r) for r in result]

    def execute_write(self, cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """写事务，返回 counters 摘要。"""
        drv = self._get_driver()
        p = params or {}

        def work(tx: Any) -> dict[str, Any]:
            res = tx.run(cypher, p)
            summary = res.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }

        with drv.session() as session:
            return session.execute_write(work)

    def ensure_user_taboo_schema(self) -> None:
        """创建唯一约束（可重复调用，已存在则忽略）。"""
        self._get_driver()
        stmts = [
            "CREATE CONSTRAINT diet_user_id IF NOT EXISTS FOR (u:DietUser) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT taboo_keyword IF NOT EXISTS FOR (t:TabooKeyword) REQUIRE t.keyword IS UNIQUE",
        ]
        for cy in stmts:
            try:
                self.execute_write(cy, {})
            except Exception:
                # 社区版/权限不足时跳过
                pass

    def merge_user_avoids_keyword(self, user_id: str, keyword: str) -> dict[str, Any]:
        """
        MERGE (:DietUser)-[:AVOIDS]->(:TabooKeyword)，用于把忌口关系落到图上。
        """
        cy = """
        MERGE (u:DietUser {id: $user_id})
        MERGE (t:TabooKeyword {keyword: $keyword})
        MERGE (u)-[:AVOIDS]->(t)
        """
        return self.execute_write(cy, {"user_id": user_id, "keyword": keyword.strip()})

    def list_user_taboos(self, user_id: str) -> list[str]:
        """列出某用户通过图边关联的忌口关键词。"""
        cy = """
        MATCH (u:DietUser {id: $user_id})-[:AVOIDS]->(t:TabooKeyword)
        RETURN t.keyword AS kw ORDER BY kw
        """
        rows = self.execute_read(cy, {"user_id": user_id})
        return [str(r.get("kw", "")).strip() for r in rows if r.get("kw")]


Neo4jStoreStub = Neo4jStore
