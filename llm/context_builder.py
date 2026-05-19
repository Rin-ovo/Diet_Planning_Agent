from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from storage.memory_store import smart_memory_file_for_current_user
from tools._util import load_profile

from .memory_digest_rank import HybridRankParams, rank_consolidated_for_digest, rank_episodes_for_digest


@dataclass
class ContextBuilderConfig:
    """控制上下文长度与自动注入片段。"""

    max_non_system_messages: int = 80
    inject_profile_digest: bool = True
    inject_smart_memory_digest: bool = True
    profile_max_list_items: int = 8
    memory_max_consolidated: int = 3
    memory_max_episodes: int = 6
    digest_max_chars: int = 3500
    # --- digest 混合排名（显著度 + 新近性 + 关键词 + 可选向量）---
    digest_use_hybrid_rank: bool = True
    digest_recency_tau_hours: float = 336.0
    digest_w_salience: float = 0.35
    digest_w_recency: float = 0.30
    digest_w_keyword: float = 0.35
    digest_w_semantic: float = 0.0
    digest_neutral_keyword: float = 0.5
    digest_pool_cap_consolidated: int = 48
    digest_pool_cap_episodes: int = 200
    # --- system 分段模板（Role / Task / Evidence / Context / Output）---
    # digest_use_structured_sections: True=分段标题；False=旧式「--- + 中文小标题」
    digest_use_structured_sections: bool = True
    # digest_structured_include_task: True 时在 system 内写 [Task]（可能与下一条 user 重复，可关）
    digest_structured_include_task: bool = True


@dataclass
class ContextBuilder:
    """
    在调用 LLM 前组装 messages：系统提示增强（档案/记忆摘要）、会话截断。
    与「记忆工程」关系：从持久化存储读取摘要并注入 system，减少模型漏调工具时的信息缺失；
    细粒度事实仍以工具为准。

    调用结构（建议）：
      1. initial_messages() → 仅 system，无本轮 query 时记忆 digest 用中性关键词分。
      2. messages_for_user_turn(session, user_text) → refresh_system_message(..., digest_query=user_text) 后追加 user。
      3. 若已有 session 且需刷新：refresh_system_message(messages, digest_query=...) 应在追加本轮 user 之前调用并传入 user_text。

    system 默认使用分段模板（config.digest_use_structured_sections）：[Role & Policies]、[Task]（可关）、
    [Evidence]（智能记忆）、[Context]（用户画像）、[Output]；关闭后回退为「--- + 中文小标题」旧格式。
    """

    base_system_prompt: str
    config: ContextBuilderConfig = field(default_factory=ContextBuilderConfig)

    def _memory_path(self):
        return smart_memory_file_for_current_user()

    def _load_smart_memory_raw(self) -> dict[str, Any] | None:
        p = self._memory_path()
        if not p.is_file():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _profile_digest(self) -> str:
        from storage.user_context import get_current_user_display, get_current_user_id

        prof = load_profile()
        lines: list[str] = [
            f"当前规划用户：{get_current_user_display()}（id={get_current_user_id()}）",
        ]

        def take_list(key: str, label: str) -> None:
            v = prof.get(key)
            if not isinstance(v, list) or not v:
                return
            cap = max(1, self.config.profile_max_list_items)
            items = [str(x).strip() for x in v if str(x).strip()][:cap]
            if items:
                lines.append(f"{label}：{', '.join(items)}")

        take_list("favorites", "喜好")
        take_list("taboos", "忌口")
        take_list("allergies", "过敏")
        try:
            from storage.neo4j_bridge import list_graph_taboo_keywords

            gt = list_graph_taboo_keywords(get_current_user_id())
            if gt:
                cap = max(1, self.config.profile_max_list_items)
                shown = gt[:cap]
                suf = "…" if len(gt) > cap else ""
                lines.append(f"图库忌口（Neo4j）: {', '.join(shown)}{suf}")
        except Exception:
            pass
        take_list("preferred_cuisines", "偏好菜系")

        g = str(prof.get("goals", "") or "").strip()
        if g:
            lines.append(f"目标：{g[:200]}{'…' if len(g) > 200 else ''}")

        notes = str(prof.get("notes", "") or "").strip()
        if notes:
            lines.append(f"备注：{notes[:200]}{'…' if len(notes) > 200 else ''}")

        stats: list[str] = []
        for k, label in (
            ("sex", "性别"),
            ("age", "年龄"),
            ("height_cm", "身高cm"),
            ("weight_kg", "体重kg"),
            ("activity_level", "活动量"),
        ):
            val = prof.get(k)
            if val is not None and str(val).strip():
                stats.append(f"{label}{val}")
        if stats:
            lines.append("体征/活动：" + "，".join(stats))

        mood = str(prof.get("current_mood", "") or "").strip()
        if mood:
            lines.append(f"当前心情：{mood}")

        return "\n".join(lines) if lines else ""

    def _hybrid_params(self) -> HybridRankParams:
        c = self.config
        return HybridRankParams(
            tau_hours=float(c.digest_recency_tau_hours),
            w_salience=float(c.digest_w_salience),
            w_recency=float(c.digest_w_recency),
            w_keyword=float(c.digest_w_keyword),
            w_semantic=float(c.digest_w_semantic),
            neutral_keyword=float(c.digest_neutral_keyword),
        )

    def _smart_memory_digest(self, digest_query: str | None) -> str:
        raw = self._load_smart_memory_raw()
        if not raw:
            return ""

        now = datetime.now(timezone.utc)
        lines: list[str] = []

        cons = [c for c in (raw.get("consolidated") or []) if isinstance(c, dict)]
        eps = [e for e in (raw.get("episodes") or []) if isinstance(e, dict)]

        def _expired(e: dict[str, Any]) -> bool:
            exp_s = e.get("expires_at")
            if not exp_s:
                return False
            try:
                exp = datetime.fromisoformat(str(exp_s).replace("Z", "+00:00"))
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                return exp < now
            except (TypeError, ValueError):
                return False

        q = (digest_query or "").strip() or None
        c = self.config
        if c.digest_use_hybrid_rank:
            hp = self._hybrid_params()
            picked_cons = rank_consolidated_for_digest(
                cons,
                query=q,
                now=now,
                params=hp,
                max_take=max(1, c.memory_max_consolidated),
                pool_cap=max(c.digest_pool_cap_consolidated, c.memory_max_consolidated),
            )
            picked_eps = rank_episodes_for_digest(
                eps,
                query=q,
                now=now,
                params=hp,
                max_take=max(1, c.memory_max_episodes),
                pool_cap=max(c.digest_pool_cap_episodes, c.memory_max_episodes),
                is_expired=_expired,
            )
        else:
            cons.sort(
                key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("updated_at", "")))
            )
            mc = c.memory_max_consolidated
            picked_cons = [cc for cc in cons[:mc] if str(cc.get("summary_text", "")).strip()]

            active = [e for e in eps if str(e.get("status", "active")) == "active"]
            active.sort(
                key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("created_at", "")))
            )
            picked_eps = []
            me = c.memory_max_episodes
            for e in active:
                if _expired(e):
                    continue
                if str(e.get("text", "")).strip():
                    picked_eps.append(e)
                if len(picked_eps) >= me:
                    break

        for con in picked_cons:
            st = str(con.get("summary_text", "")).strip()
            if not st:
                continue
            if len(st) > 600:
                st = st[:597] + "..."
            lines.append(f"[摘要] {st}")

        for e in picked_eps:
            t = str(e.get("text", "")).strip()
            if not t:
                continue
            tags = e.get("tags")
            tag_s = ""
            if isinstance(tags, list) and tags:
                tag_s = f" ({','.join(str(x) for x in tags[:4])})"
            one = t.replace("\n", " ")
            if len(one) > 220:
                one = one[:217] + "..."
            lines.append(f"[片段{tag_s}] {one}")

        body = "\n".join(lines)
        if len(body) > c.digest_max_chars:
            body = body[: c.digest_max_chars - 1] + "…"
        return body

    def _structure_system_content(
        self,
        *,
        role_policies: str,
        profile_digest: str | None,
        memory_digest: str | None,
        user_query: str | None,
    ) -> str:
        """将角色规则、画像、记忆与（可选）当前提问组织为分段式 system（对齐 RAG 常用模板）。"""
        sections: list[str] = []
        rp = (role_policies or "").strip()
        if rp:
            sections.append("[Role & Policies]\n" + rp)

        tq = (user_query or "").strip()
        if self.config.digest_structured_include_task and tq:
            sections.append("[Task]\n" + tq)

        mem = (memory_digest or "").strip()
        if mem:
            sections.append(
                "[Evidence]\n"
                "（以下为自动注入的智能记忆摘要；完整与写入请用 smart_memory_* 工具。）\n"
                + mem
            )

        prof = (profile_digest or "").strip()
        if prof:
            sections.append(
                "[Context]\n"
                "（以下为自动注入的用户画像摘要；完整与更新请用 get_user_profile / update_user_profile。）\n"
                + prof
            )

        sections.append(
            "[Output]\n"
            "请基于以上角色与规则、用户画像与智能记忆，必要时调用工具核对或更新数据，用自然中文给出有据的回答。"
        )
        return "\n\n".join(sections)

    def build_system_content(self, *, digest_query: str | None = None) -> str:
        """拼接最终 system：基础提示 + 可选档案/记忆摘要。digest_query 为本轮用户话，用于记忆混合排名。"""
        base = self.base_system_prompt.rstrip()

        profile_block: str | None = None
        if self.config.inject_profile_digest:
            d = self._profile_digest()
            if d:
                profile_block = d

        memory_block: str | None = None
        if self.config.inject_smart_memory_digest:
            m = self._smart_memory_digest(digest_query)
            if m:
                memory_block = m

        if self.config.digest_use_structured_sections:
            return self._structure_system_content(
                role_policies=base,
                profile_digest=profile_block,
                memory_digest=memory_block,
                user_query=digest_query,
            )

        parts: list[str] = [base]
        if profile_block:
            parts.append(
                "\n\n---\n【档案摘要（自动注入；完整与更新请用 get_user_profile / update_user_profile）】\n"
                + profile_block
            )
        if memory_block:
            parts.append(
                "\n\n---\n【智能记忆摘要（自动注入；检索/写入请用 smart_memory_* 工具）】\n"
                + memory_block
            )
        return "".join(parts)

    def refresh_system_message(
        self,
        messages: list[dict[str, Any]],
        *,
        digest_query: str | None = None,
    ) -> None:
        """就地更新首条 system；digest_query 建议传本轮用户输入（更稳妥），勿依赖尚未 append 的 user。"""
        if not messages:
            return
        if messages[0].get("role") != "system":
            return
        messages[0] = {
            "role": "system",
            "content": self.build_system_content(digest_query=digest_query),
        }

    def initial_messages(self) -> list[dict[str, Any]]:
        """新会话：仅含一条增强后的 system（无本轮 query，记忆 digest 中关键词分为中性）。"""
        return [{"role": "system", "content": self.build_system_content(digest_query=None)}]

    def trim_messages(self, messages: list[dict[str, Any]]) -> None:
        """
        保留首条 system，其余仅保留最近 max_non_system_messages 条（含 user/assistant/tool）。
        就地修改列表。
        """
        cap = max(10, int(self.config.max_non_system_messages))
        if len(messages) <= 1 + cap:
            return
        head = messages[0]
        tail = messages[-cap:]
        messages.clear()
        messages.append(head)
        messages.extend(tail)

    def messages_for_user_turn(
        self,
        session_messages: list[dict[str, Any]],
        user_text: str,
    ) -> list[dict[str, Any]]:
        """调用图之前：用本轮 user_text 刷新 system（含记忆混合排名）+ 追加 user。"""
        session_messages = list(session_messages)
        self.refresh_system_message(session_messages, digest_query=user_text)
        return session_messages + [{"role": "user", "content": user_text}]


__all__ = ["ContextBuilder", "ContextBuilderConfig"]
