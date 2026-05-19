"""
智能记忆 digest 的混合打分：显著度 + 新近性 + 关键词相关性 +（可选）向量相关性。

无进程内缓存；向量路径仅调用 storage.smart_memory_faiss 的嵌入（内部 ST 单例，与 FAISS 索引缓存无关）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

# 每条 episode 参与关键词/向量打分的最大字符（控制编码量）
_EP_TEXT_PREVIEW = 1200
_SUM_PREVIEW = 2000


def _parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _norm_salience(v: Any) -> float:
    try:
        x = float(v or 1.0)
    except (TypeError, ValueError):
        x = 1.0
    x = max(0.05, min(2.0, x))
    return (x - 0.05) / (2.0 - 0.05)


def _recency_score(ts: Any, now: datetime, tau_hours: float) -> float:
    dt = _parse_iso(ts)
    if dt is None:
        return 0.35
    delta = now - dt
    hours = max(0.0, delta.total_seconds() / 3600.0)
    tau = max(1.0, float(tau_hours))
    return float(math.exp(-hours / tau))


def _keyword_relevance(query: str, text: str, tags: Any, *, neutral: float) -> float:
    q = (query or "").strip().lower()
    if not q:
        return neutral
    hay = (text or "").lower()
    tag_parts: list[str] = []
    if isinstance(tags, list):
        tag_parts = [str(t).lower() for t in tags if str(t).strip()]
    combined = hay + " " + " ".join(tag_parts)
    if q in combined:
        return 1.0
    parts = [p for p in re.split(r"\s+", q) if len(p) >= 2]
    if not parts:
        parts = [q] if len(q) >= 1 else []
    if not parts:
        return neutral
    hit = sum(1 for p in parts if p in combined)
    base = hit / len(parts)
    set_q = set(q)
    set_c = set(combined)
    overlap = len(set_q & set_c) / max(len(set_q), 1) if set_q else 0.0
    return min(1.0, max(base, overlap * 0.85))


def _semantic_scores(
    query: str,
    texts: list[str],
    *,
    neutral: float,
) -> list[float]:
    n = len(texts)
    if n == 0:
        return []
    q = (query or "").strip()
    if not q:
        return [neutral] * n
    try:
        from storage.smart_memory_faiss import digest_embedding_available, encode_digest_texts
    except ImportError:
        return [neutral] * n
    if not digest_embedding_available():
        return [neutral] * n

    vecs = encode_digest_texts(texts)
    qv = encode_digest_texts([q[:8000]])
    if vecs is None or qv is None or vecs.shape[0] != n:
        return [neutral] * n
    q1 = qv.reshape(-1)
    dots = vecs @ q1
    out: list[float] = []
    for i in range(n):
        c = float(dots[i])
        out.append(min(1.0, max(0.0, (c + 1.0) / 2.0)))
    return out


@dataclass
class HybridRankParams:
    tau_hours: float
    w_salience: float
    w_recency: float
    w_keyword: float
    w_semantic: float
    neutral_keyword: float


def _normalize_weights(p: HybridRankParams) -> tuple[float, float, float, float]:
    ws = max(0.0, p.w_salience)
    wr = max(0.0, p.w_recency)
    wk = max(0.0, p.w_keyword)
    we = max(0.0, p.w_semantic)
    tot = ws + wr + wk + we
    if tot <= 0:
        return 1.0, 0.0, 0.0, 0.0
    return ws / tot, wr / tot, wk / tot, we / tot


def _hybrid_score(
    sal: float,
    rec: float,
    kw: float,
    sem: float,
    p: HybridRankParams,
) -> float:
    ws, wr, wk, we = _normalize_weights(p)
    return ws * sal + wr * rec + wk * kw + we * sem


def rank_consolidated_for_digest(
    items: list[dict[str, Any]],
    *,
    query: str | None,
    now: datetime,
    params: HybridRankParams,
    max_take: int,
    pool_cap: int,
) -> list[dict[str, Any]]:
    pool = [c for c in items if isinstance(c, dict) and str(c.get("summary_text", "")).strip()]
    pool.sort(
        key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("updated_at", ""))),
    )
    pool = pool[: max(pool_cap, max_take)]

    texts = [str(c.get("summary_text", "")).strip()[:_SUM_PREVIEW] for c in pool]
    sems = _semantic_scores(query or "", texts, neutral=params.neutral_keyword) if params.w_semantic > 0 else [params.neutral_keyword] * len(pool)

    scored: list[tuple[float, dict[str, Any]]] = []
    for i, c in enumerate(pool):
        sal = _norm_salience(c.get("salience", 1))
        rec = _recency_score(c.get("updated_at") or c.get("created_at"), now, params.tau_hours)
        kw = _keyword_relevance(
            query or "",
            str(c.get("summary_text", "")),
            c.get("tags"),
            neutral=params.neutral_keyword,
        )
        sem = sems[i] if i < len(sems) else params.neutral_keyword
        sc = _hybrid_score(sal, rec, kw, sem, params)
        scored.append((sc, c))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
    return [c for _, c in scored[:max_take]]


def rank_episodes_for_digest(
    items: list[dict[str, Any]],
    *,
    query: str | None,
    now: datetime,
    params: HybridRankParams,
    max_take: int,
    pool_cap: int,
    is_expired: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    active = [
        e
        for e in items
        if isinstance(e, dict)
        and str(e.get("status", "active")) == "active"
        and not is_expired(e)
        and str(e.get("text", "")).strip()
    ]
    active.sort(
        key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("created_at", ""))),
    )
    pool = active[: max(pool_cap, max_take)]

    texts = [str(e.get("text", "")).strip()[:_EP_TEXT_PREVIEW] for e in pool]
    sems = _semantic_scores(query or "", texts, neutral=params.neutral_keyword) if params.w_semantic > 0 else [params.neutral_keyword] * len(pool)

    scored: list[tuple[float, dict[str, Any]]] = []
    for i, e in enumerate(pool):
        sal = _norm_salience(e.get("salience", 1))
        rec = _recency_score(e.get("updated_at") or e.get("created_at"), now, params.tau_hours)
        kw = _keyword_relevance(query or "", str(e.get("text", "")), e.get("tags"), neutral=params.neutral_keyword)
        sem = sems[i] if i < len(sems) else params.neutral_keyword
        sc = _hybrid_score(sal, rec, kw, sem, params)
        scored.append((sc, e))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("created_at", "")), str(x[1].get("id", ""))))
    return [e for _, e in scored[:max_take]]
