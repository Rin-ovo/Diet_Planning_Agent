from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from storage.memory_store import smart_memory_file_for_current_user
from storage.smart_memory_faiss import semantic_index_stats, semantic_search

from ._util import dumps, err

_STORE_VERSION = 1
_MAX_EPISODES = 1200
_MAX_CONSOLIDATED = 120
_DEFAULT_SALIENCE = 1.0


def _memory_path():
    return smart_memory_file_for_current_user()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _default_store() -> dict[str, Any]:
    return {
        "version": _STORE_VERSION,
        "episodes": [],
        "consolidated": [],
        "meta": {"last_vacuum_at": "", "last_summarize_at": ""},
    }


def _load_store() -> dict[str, Any]:
    p = _memory_path()
    if not p.is_file():
        data = _default_store()
        _save_store(data)
        return data

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return _default_store()
    base = _default_store()
    base.update(raw)
    if not isinstance(base.get("episodes"), list):
        base["episodes"] = []
    if not isinstance(base.get("consolidated"), list):
        base["consolidated"] = []
    if not isinstance(base.get("meta"), dict):
        base["meta"] = {"last_vacuum_at": "", "last_summarize_at": ""}
    return base


def _save_store(data: dict[str, Any]) -> None:
    p = _memory_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _as_tags(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        t = v.strip()
        return [t] if t else []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    return []


def _episode_active(e: dict[str, Any]) -> bool:
    return str(e.get("status", "active")) == "active"


def _episode_expired(e: dict[str, Any], now: datetime) -> bool:
    exp = _parse_iso(e.get("expires_at"))
    if exp is None:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp < now


def _episode_matches_retrieve_filters(
    e: dict[str, Any],
    *,
    query: str,
    tags_any: list[str],
    include_archived: bool,
    now: datetime,
) -> bool:
    if not include_archived and not _episode_active(e):
        return False
    if _episode_expired(e, now) and not include_archived:
        return False
    text = str(e.get("text", "")).lower()
    etags = [str(t).lower() for t in _as_tags(e.get("tags"))]
    if tags_any:
        if not any(t in etags for t in [x.lower() for x in tags_any]):
            return False
    if query:
        if query not in text and not any(query in t for t in etags):
            return False
    return True


def _episode_matches_semantic_filters(
    e: dict[str, Any],
    *,
    tags_any: list[str],
    include_archived: bool,
    now: datetime,
) -> bool:
    """语义召回命中后：仅校验活跃/过期与标签，不要求 query 子串命中。"""
    if not include_archived and not _episode_active(e):
        return False
    if _episode_expired(e, now) and not include_archived:
        return False
    if tags_any:
        etags = [str(t).lower() for t in _as_tags(e.get("tags"))]
        if not any(t in etags for t in [x.lower() for x in tags_any]):
            return False
    return True


def _consolidated_matches_semantic_filters(
    c: dict[str, Any],
    *,
    tags_any: list[str],
) -> bool:
    if tags_any:
        ctags = [str(t).lower() for t in _as_tags(c.get("tags"))]
        if not any(t in ctags for t in [x.lower() for x in tags_any]):
            return False
    return True


def _consolidated_matches_retrieve_filters(
    c: dict[str, Any],
    *,
    query: str,
    tags_any: list[str],
) -> bool:
    st = str(c.get("summary_text", "")).lower()
    ctags = [str(t).lower() for t in _as_tags(c.get("tags"))]
    if tags_any and not any(t in ctags for t in [x.lower() for x in tags_any]):
        return False
    if query and query not in st and not any(query in t for t in ctags):
        return False
    return True


# --- 创建 ---


def tool_smart_memory_create(params: dict[str, Any]) -> str:
    text = str(params.get("text", "")).strip()
    if not text:
        return err("text 不能为空：请写入可复用的对话要点（偏好、反感、目标变化等）")
    if len(text) > 8000:
        return err("text 过长，请压缩到 8000 字以内")

    tags = _as_tags(params.get("tags"))
    source = str(params.get("source", "dialogue")).strip() or "dialogue"
    salience = float(params.get("salience", _DEFAULT_SALIENCE) or _DEFAULT_SALIENCE)
    salience = max(0.05, min(2.0, salience))

    ttl_days = params.get("ttl_days")
    expires_at: str | None = None
    if ttl_days is not None:
        try:
            d = int(ttl_days)
            if d > 0:
                expires_at = (datetime.now(timezone.utc) + timedelta(days=d)).isoformat()
        except (TypeError, ValueError):
            pass

    store = _load_store()
    eps: list[dict[str, Any]] = list(store.get("episodes") or [])
    now = _now_iso()
    ep = {
        "id": _new_id("ep"),
        "created_at": now,
        "updated_at": now,
        "text": text,
        "source": source,
        "tags": tags,
        "salience": salience,
        "expires_at": expires_at,
        "status": "active",
    }
    eps.append(ep)
    store["episodes"] = eps
    _vacuum_store(store, soft=True)
    _save_store(store)
    return dumps(
        {
            "ok": True,
            "id": ep["id"],
            "expires_at": expires_at,
            "hint": "后续可用 smart_memory_retrieve（关键词/标签 + 可选 BGE-M3+FAISS 语义）；定期 smart_memory_summarize；过期用 smart_memory_forget。",
        }
    )


# --- 检索 ---


def tool_smart_memory_retrieve(params: dict[str, Any]) -> str:
    query_raw = str(params.get("query", "")).strip()
    query = query_raw.lower()
    tags_any = _as_tags(params.get("tags_any"))
    try:
        limit = int(params.get("limit", 12) or 12)
    except (TypeError, ValueError):
        limit = 12
    limit = max(1, min(40, limit))

    uf = params.get("use_faiss", True)
    if isinstance(uf, str):
        use_faiss = uf.strip().lower() not in ("0", "false", "no", "off")
    else:
        use_faiss = bool(uf)

    include_archived = bool(params.get("include_archived", False))
    include_consolidated = bool(params.get("include_consolidated", True))

    store = _load_store()
    now = datetime.now(timezone.utc)
    eps: list[dict[str, Any]] = list(store.get("episodes") or [])

    candidates: list[dict[str, Any]] = []
    for e in eps:
        if not isinstance(e, dict):
            continue
        if not _episode_matches_retrieve_filters(
            e, query=query, tags_any=tags_any, include_archived=include_archived, now=now
        ):
            continue
        candidates.append(e)

    def sort_key(x: dict[str, Any]) -> tuple[float, str]:
        sal = float(x.get("salience", _DEFAULT_SALIENCE) or _DEFAULT_SALIENCE)
        ca = str(x.get("created_at", ""))
        return (-sal, ca)

    candidates.sort(key=sort_key)

    by_ep_id: dict[str, dict[str, Any]] = {}
    for e in eps:
        if isinstance(e, dict) and e.get("id"):
            by_ep_id[str(e["id"])] = e

    vec_ep_ids: list[tuple[str, float]] = []
    vec_cons_ids: list[tuple[str, float]] = []
    faiss_semantic = False
    if use_faiss and query_raw:
        try:
            hits = semantic_search(
                _memory_path(),
                store,
                query_raw,
                top_k=max(limit, 24),
            )
        except Exception:
            hits = []
        if hits:
            faiss_semantic = True
            for kind, rid, sc in hits:
                if kind == "episode":
                    vec_ep_ids.append((rid, sc))
                elif kind == "consolidated":
                    vec_cons_ids.append((rid, sc))

    merged_ep: list[str] = []
    seen_ep: set[str] = set()
    for eid, _ in vec_ep_ids:
        if eid in seen_ep or eid not in by_ep_id:
            continue
        ev = by_ep_id[eid]
        if not _episode_matches_semantic_filters(
            ev, tags_any=tags_any, include_archived=include_archived, now=now
        ):
            continue
        merged_ep.append(eid)
        seen_ep.add(eid)
    for e in candidates:
        eid = str(e.get("id", ""))
        if not eid or eid in seen_ep:
            continue
        merged_ep.append(eid)
        seen_ep.add(eid)
        if len(merged_ep) >= limit * 3:
            break

    picked = [by_ep_id[i] for i in merged_ep[:limit] if i in by_ep_id]

    out_cons: list[dict[str, Any]] = []
    if include_consolidated:
        cons_all: list[dict[str, Any]] = [c for c in (store.get("consolidated") or []) if isinstance(c, dict)]
        by_cid = {str(c.get("id")): c for c in cons_all if c.get("id")}

        merged_c: list[str] = []
        seen_c: set[str] = set()
        for cid, _ in vec_cons_ids:
            if cid in seen_c or cid not in by_cid:
                continue
            c = by_cid[cid]
            if not _consolidated_matches_semantic_filters(c, tags_any=tags_any):
                continue
            merged_c.append(cid)
            seen_c.add(cid)

        kw_cons: list[dict[str, Any]] = []
        for c in cons_all:
            if not _consolidated_matches_retrieve_filters(c, query=query, tags_any=tags_any):
                continue
            kw_cons.append(c)
        kw_cons.sort(
            key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("updated_at", "")))
        )
        for c in kw_cons:
            cid = str(c.get("id", ""))
            if not cid or cid in seen_c:
                continue
            merged_c.append(cid)
            seen_c.add(cid)
            if len(merged_c) >= max(1, limit // 2) * 3:
                break

        cap_c = max(1, limit // 2)
        out_cons = [by_cid[i] for i in merged_c[:cap_c] if i in by_cid]

    return dumps(
        {
            "query": query or None,
            "tags_any": tags_any or None,
            "faiss_semantic": faiss_semantic,
            "episodes": [
                {
                    "id": e.get("id"),
                    "text": e.get("text"),
                    "tags": e.get("tags"),
                    "salience": e.get("salience"),
                    "source": e.get("source"),
                    "status": e.get("status"),
                    "created_at": e.get("created_at"),
                    "expires_at": e.get("expires_at"),
                }
                for e in picked
            ],
            "consolidated": [
                {
                    "id": c.get("id"),
                    "summary_text": c.get("summary_text"),
                    "tags": c.get("tags"),
                    "salience": c.get("salience"),
                    "source_episode_ids": c.get("source_episode_ids"),
                    "updated_at": c.get("updated_at"),
                }
                for c in out_cons
            ],
        }
    )


# --- 摘要与整合（无二次模型：规则拼接 + 可选归档源片段） ---


def tool_smart_memory_summarize(params: dict[str, Any]) -> str:
    tag = str(params.get("tag", "")).strip()
    try:
        max_episodes = int(params.get("max_episodes", 10) or 10)
    except (TypeError, ValueError):
        max_episodes = 10
    max_episodes = max(2, min(30, max_episodes))

    archive_sources = bool(params.get("archive_sources", True))
    title = str(params.get("title", "")).strip() or "对话偏好摘要"

    store = _load_store()
    now = datetime.now(timezone.utc)
    eps: list[dict[str, Any]] = list(store.get("episodes") or [])

    pool: list[dict[str, Any]] = []
    for e in eps:
        if not isinstance(e, dict):
            continue
        if not _episode_active(e):
            continue
        if _episode_expired(e, now):
            continue
        if tag:
            etags = [str(t).lower() for t in _as_tags(e.get("tags"))]
            if tag.lower() not in etags:
                continue
        pool.append(e)

    pool.sort(
        key=lambda x: (-float(x.get("salience", 1) or 1), str(x.get("created_at", "")))
    )
    chosen = pool[:max_episodes]
    if len(chosen) < 1:
        return err("没有可摘要的活跃记忆：先 smart_memory_create，或放宽 tag")

    lines: list[str] = []
    src_ids: list[str] = []
    merge_tags: set[str] = set()
    for e in chosen:
        tid = str(e.get("id", ""))
        if tid:
            src_ids.append(tid)
        t = str(e.get("text", "")).strip()
        if t:
            one = re.sub(r"\s+", " ", t)
            if len(one) > 280:
                one = one[:277] + "..."
            lines.append(f"- {one}")
        for tg in _as_tags(e.get("tags")):
            merge_tags.add(tg)

    summary_text = f"【{title}】\n" + "\n".join(lines)
    if len(summary_text) > 12000:
        summary_text = summary_text[:11997] + "..."

    cons_list: list[dict[str, Any]] = list(store.get("consolidated") or [])
    cid = _new_id("sum")
    tnow = _now_iso()
    cons = {
        "id": cid,
        "created_at": tnow,
        "updated_at": tnow,
        "title": title,
        "summary_text": summary_text,
        "source_episode_ids": src_ids,
        "tags": sorted(merge_tags),
        "salience": min(
            2.0,
            max(float(x.get("salience", 1) or 1) for x in chosen) + 0.1,
        ),
    }
    cons_list.append(cons)
    store["consolidated"] = cons_list

    if archive_sources:
        idset = set(src_ids)
        for e in eps:
            if str(e.get("id")) in idset:
                e["status"] = "archived"
                e["updated_at"] = tnow
        store["episodes"] = eps

    meta = store.get("meta") if isinstance(store.get("meta"), dict) else {}
    meta["last_summarize_at"] = tnow
    store["meta"] = meta

    _vacuum_store(store, soft=True)
    _save_store(store)
    return dumps(
        {
            "ok": True,
            "consolidated_id": cid,
            "covered_episode_ids": src_ids,
            "archived_sources": archive_sources,
            "preview": summary_text[:500] + ("..." if len(summary_text) > 500 else ""),
        }
    )


# --- 遗忘 / 衰减 ---


def tool_smart_memory_forget(params: dict[str, Any]) -> str:
    mode = str(params.get("mode", "")).strip().lower()
    store = _load_store()
    now = datetime.now(timezone.utc)
    tnow = _now_iso()

    if mode == "delete_episode":
        eid = str(params.get("id", "")).strip()
        if not eid:
            return err("delete_episode 需要 id")
        eps = [e for e in store.get("episodes") or [] if isinstance(e, dict) and str(e.get("id")) != eid]
        store["episodes"] = eps
        _save_store(store)
        return dumps({"ok": True, "removed_episode": eid})

    if mode == "delete_consolidated":
        cid = str(params.get("id", "")).strip()
        if not cid:
            return err("delete_consolidated 需要 id")
        cons = [c for c in store.get("consolidated") or [] if isinstance(c, dict) and str(c.get("id")) != cid]
        store["consolidated"] = cons
        _save_store(store)
        return dumps({"ok": True, "removed_consolidated": cid})

    if mode == "archive_episode":
        eid = str(params.get("id", "")).strip()
        if not eid:
            return err("archive_episode 需要 id")
        n = 0
        for e in store.get("episodes") or []:
            if isinstance(e, dict) and str(e.get("id")) == eid:
                e["status"] = "archived"
                e["updated_at"] = tnow
                n += 1
        _save_store(store)
        return dumps({"ok": True, "archived": n})

    if mode == "decay_episode":
        eid = str(params.get("id", "")).strip()
        if not eid:
            return err("decay_episode 需要 id")
        try:
            factor = float(params.get("factor", 0.65) or 0.65)
        except (TypeError, ValueError):
            factor = 0.65
        factor = max(0.05, min(1.0, factor))
        n = 0
        for e in store.get("episodes") or []:
            if isinstance(e, dict) and str(e.get("id")) == eid:
                old = float(e.get("salience", _DEFAULT_SALIENCE) or _DEFAULT_SALIENCE)
                e["salience"] = round(max(0.05, old * factor), 4)
                e["updated_at"] = tnow
                n += 1
        _save_store(store)
        return dumps({"ok": True, "decayed": n, "factor": factor})

    if mode == "prune_expired":
        removed = 0
        kept: list[dict[str, Any]] = []
        for e in store.get("episodes") or []:
            if not isinstance(e, dict):
                continue
            if _episode_expired(e, now):
                removed += 1
                continue
            kept.append(e)
        store["episodes"] = kept
        _save_store(store)
        return dumps({"ok": True, "removed_expired_episodes": removed})

    if mode == "prune_low_salience":
        try:
            threshold = float(params.get("min_salience", 0.12) or 0.12)
        except (TypeError, ValueError):
            threshold = 0.12
        removed = 0
        kept = []
        for e in store.get("episodes") or []:
            if not isinstance(e, dict):
                continue
            sal = float(e.get("salience", _DEFAULT_SALIENCE) or _DEFAULT_SALIENCE)
            if _episode_active(e) and sal < threshold:
                removed += 1
                continue
            kept.append(e)
        store["episodes"] = kept
        _save_store(store)
        return dumps({"ok": True, "removed_below_salience": removed, "threshold": threshold})

    if mode == "decay_consolidated":
        cid = str(params.get("id", "")).strip()
        try:
            factor = float(params.get("factor", 0.7) or 0.7)
        except (TypeError, ValueError):
            factor = 0.7
        factor = max(0.05, min(1.0, factor))
        n = 0
        for c in store.get("consolidated") or []:
            if isinstance(c, dict) and (not cid or str(c.get("id")) == cid):
                old = float(c.get("salience", 1) or 1)
                c["salience"] = round(max(0.05, old * factor), 4)
                c["updated_at"] = tnow
                n += 1
        if cid and n == 0:
            return err("未找到该 consolidated id")
        _save_store(store)
        return dumps({"ok": True, "decayed_consolidated": n, "factor": factor})

    return err(
        "mode 须为: delete_episode | delete_consolidated | archive_episode | "
        "decay_episode | decay_consolidated | prune_expired | prune_low_salience"
    )


# --- 合并（整合多条 episode 为一条，源可归档） ---


def tool_smart_memory_merge(params: dict[str, Any]) -> str:
    ids = params.get("source_ids")
    if not isinstance(ids, list) or len(ids) < 2:
        return err("source_ids 须为至少 2 个字符串 id")
    id_list = [str(x).strip() for x in ids if str(x).strip()]
    if len(id_list) < 2:
        return err("source_ids 有效项不足 2")

    merged_text = str(params.get("merged_text", "")).strip()
    new_tags = _as_tags(params.get("tags"))
    archive_sources = bool(params.get("archive_sources", True))

    store = _load_store()
    eps: list[dict[str, Any]] = list(store.get("episodes") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for e in eps:
        if isinstance(e, dict) and e.get("id"):
            by_id[str(e["id"])] = e

    missing = [i for i in id_list if i not in by_id]
    if missing:
        return err(f"找不到 episode id: {missing}")

    parts: list[str] = []
    tag_set = set(new_tags)
    for i in id_list:
        e = by_id[i]
        parts.append(str(e.get("text", "")).strip())
        for tg in _as_tags(e.get("tags")):
            tag_set.add(tg)

    body = merged_text if merged_text else "\n".join(p for p in parts if p)
    if not body.strip():
        return err("合并后正文为空")

    try:
        salience = float(params.get("salience"))
    except (TypeError, ValueError):
        salience = max(
            float(by_id[i].get("salience", 1) or 1) for i in id_list
        )
    salience = max(0.05, min(2.0, salience + 0.05))

    now = _now_iso()
    new_ep = {
        "id": _new_id("ep"),
        "created_at": now,
        "updated_at": now,
        "text": body[:8000],
        "source": "merged",
        "tags": sorted(tag_set),
        "salience": salience,
        "expires_at": None,
        "status": "active",
        "merged_from_ids": id_list,
    }
    eps.append(new_ep)

    if archive_sources:
        idset = set(id_list)
        for e in eps:
            if str(e.get("id")) in idset:
                e["status"] = "archived"
                e["updated_at"] = now

    store["episodes"] = eps
    _vacuum_store(store, soft=True)
    _save_store(store)
    return dumps(
        {
            "ok": True,
            "new_episode_id": new_ep["id"],
            "archived_sources": archive_sources,
        }
    )


# --- 管理：统计 / 最近列表 / 压缩容量 ---


def _vacuum_store(store: dict[str, Any], *, soft: bool) -> None:
    """控制条目数量：优先丢 archived，再按 salience+时间裁 episdoes。"""
    eps: list[dict[str, Any]] = [e for e in store.get("episodes") or [] if isinstance(e, dict)]
    cons: list[dict[str, Any]] = [c for c in store.get("consolidated") or [] if isinstance(c, dict)]

    if len(cons) > _MAX_CONSOLIDATED:
        cons.sort(key=lambda x: (float(x.get("salience", 1) or 1), str(x.get("updated_at", ""))))
        cons = cons[-_MAX_CONSOLIDATED:]

    if len(eps) > _MAX_EPISODES:
        active = [e for e in eps if _episode_active(e)]
        archived = [e for e in eps if not _episode_active(e)]
        archived.sort(key=lambda x: str(x.get("updated_at", x.get("created_at", ""))))
        drop_arch = len(eps) - _MAX_EPISODES
        if drop_arch > 0 and archived:
            archived = archived[drop_arch:]
        merged_list = active + archived
        if len(merged_list) > _MAX_EPISODES:
            merged_list.sort(
                key=lambda x: (
                    float(x.get("salience", 0.1) or 0.1),
                    str(x.get("created_at", "")),
                )
            )
            merged_list = merged_list[-_MAX_EPISODES:]
        eps = merged_list

    store["episodes"] = eps
    store["consolidated"] = cons
    if not soft:
        meta = store.get("meta") if isinstance(store.get("meta"), dict) else {}
        meta["last_vacuum_at"] = _now_iso()
        store["meta"] = meta


def tool_smart_memory_manage(params: dict[str, Any]) -> str:
    action = str(params.get("action", "stats")).strip().lower()
    store = _load_store()

    if action == "stats":
        eps = [e for e in store.get("episodes") or [] if isinstance(e, dict)]
        cons = [c for c in store.get("consolidated") or [] if isinstance(c, dict)]
        now = datetime.now(timezone.utc)
        active = sum(1 for e in eps if _episode_active(e) and not _episode_expired(e, now))
        archived = sum(1 for e in eps if not _episode_active(e))
        expired_pending = sum(1 for e in eps if _episode_expired(e, now))
        faiss_info = semantic_index_stats(_memory_path(), store)
        return dumps(
            {
                "episodes_total": len(eps),
                "episodes_active_not_expired": active,
                "episodes_archived": archived,
                "episodes_expired": expired_pending,
                "consolidated_total": len(cons),
                "caps": {"max_episodes": _MAX_EPISODES, "max_consolidated": _MAX_CONSOLIDATED},
                "meta": store.get("meta"),
                "faiss_semantic": faiss_info,
            }
        )

    if action == "list_recent":
        try:
            k = int(params.get("k", 15) or 15)
        except (TypeError, ValueError):
            k = 15
        k = max(1, min(50, k))
        eps = [e for e in store.get("episodes") or [] if isinstance(e, dict)]
        eps.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return dumps(
            {
                "episodes": [
                    {
                        "id": e.get("id"),
                        "text": (str(e.get("text", ""))[:200] + "…")
                        if len(str(e.get("text", ""))) > 200
                        else e.get("text"),
                        "tags": e.get("tags"),
                        "salience": e.get("salience"),
                        "status": e.get("status"),
                        "created_at": e.get("created_at"),
                    }
                    for e in eps[:k]
                ]
            }
        )

    if action == "vacuum":
        _vacuum_store(store, soft=False)
        _save_store(store)
        return dumps(
            {
                "ok": True,
                "episodes": len(store.get("episodes") or []),
                "consolidated": len(store.get("consolidated") or []),
            }
        )

    return err("action 须为 stats | list_recent | vacuum")


__all__ = [
    "tool_smart_memory_create",
    "tool_smart_memory_retrieve",
    "tool_smart_memory_summarize",
    "tool_smart_memory_forget",
    "tool_smart_memory_merge",
    "tool_smart_memory_manage",
]
