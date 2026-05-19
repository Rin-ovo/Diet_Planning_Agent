"""
smart_memory.json 的本地语义索引：BAAI/bge-m3 + FAISS（IndexFlatIP + L2 归一化后等价余弦）。

依赖（可选）: pip install faiss-cpu sentence-transformers
环境:
  SMART_MEMORY_FAISS   默认 1；设为 0 关闭语义召回
  LOCAL_EMBEDDING_MODEL  默认 BAAI/bge-m3
  LOCAL_EMBEDDING_DEVICE 默认 cpu

索引文件与 smart_memory.json 同目录：*.semantic.faiss、*.semantic.meta.json
当 json 的 mtime 新于 meta 中记录时自动重建索引。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_JSON_MTIME_EPS = 1e-3

_HAS_FAISS = False
_HAS_ST = False
try:
    import faiss  # type: ignore[import-not-found]

    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[misc, assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    _HAS_ST = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]

_DEFAULT_MODEL = "BAAI/bge-m3"
_ENCODE_MAX_CHARS = 8000

_bundle_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_st_model_cache: Any = None
_st_model_name: str = ""


def _using_faiss_semantic() -> bool:
    raw = (os.getenv("SMART_MEMORY_FAISS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def semantic_deps_available() -> bool:
    return bool(_HAS_FAISS and _HAS_ST)


def digest_embedding_available() -> bool:
    """ContextBuilder digest 的向量分仅依赖 sentence-transformers，不要求 faiss。"""
    return bool(_HAS_ST)


def encode_digest_texts(texts: list[str]) -> Any | None:
    """
    多条文本的 L2 归一化嵌入矩阵 (n, dim)，供 digest 混合打分。
    复用 _load_st_model 单例；不读写 FAISS 索引、不触碰 _bundle_cache。
    """
    if not _HAS_ST or not texts:
        return None
    try:
        model = _load_st_model()
        return _encode_batch(model, texts)
    except Exception:
        return None


def semantic_index_paths(memory_json: Path) -> tuple[Path, Path]:
    """与 smart_memory.json 同 stem，后缀为 .semantic.faiss / .semantic.meta.json"""
    stem = memory_json.stem
    parent = memory_json.parent
    return parent / f"{stem}.semantic.faiss", parent / f"{stem}.semantic.meta.json"


def invalidate_semantic_index(memory_json: Path) -> None:
    """删除磁盘上的语义索引（下次检索会重建）。"""
    ip, mp = semantic_index_paths(memory_json)
    for p in (ip, mp):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
    key = str(memory_json.resolve())
    _bundle_cache.pop(key, None)


def _json_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_st_model() -> Any:
    global _st_model_cache, _st_model_name
    name = (os.getenv("LOCAL_EMBEDDING_MODEL") or _DEFAULT_MODEL).strip()
    device = (os.getenv("LOCAL_EMBEDDING_DEVICE") or "cpu").strip() or "cpu"
    if SentenceTransformer is None:
        raise RuntimeError("未安装 sentence-transformers")
    if _st_model_cache is not None and _st_model_name == f"{name}\x00{device}":
        return _st_model_cache
    _st_model_cache = SentenceTransformer(name, device=device)
    _st_model_name = f"{name}\x00{device}"
    return _st_model_cache


def _encode_batch(model: Any, texts: list[str]) -> Any:
    import numpy as np

    trimmed = [(t or "")[:_ENCODE_MAX_CHARS] for t in texts]
    try:
        vecs = model.encode(
            trimmed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except TypeError:
        vecs = model.encode(trimmed, normalize_embeddings=True, convert_to_numpy=True)
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    return vecs


def _encode_query(model: Any, text: str) -> Any:
    import numpy as np

    q = _encode_batch(model, [text])
    if q.ndim == 1:
        q = q.reshape(1, -1)
    return q.astype(np.float32)


def _rows_and_texts_from_store(store: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """每条一行：episode / consolidated，用于建索引。"""
    rows: list[dict[str, str]] = []
    texts: list[str] = []

    for e in store.get("episodes") or []:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id", "")).strip()
        t = str(e.get("text", "")).strip()
        if not eid or not t:
            continue
        rows.append({"kind": "episode", "id": eid})
        texts.append(t)

    for c in store.get("consolidated") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        t = str(c.get("summary_text", "")).strip()
        if not cid or not t:
            continue
        rows.append({"kind": "consolidated", "id": cid})
        texts.append(t)

    return rows, texts


def _build_faiss_index(vectors: Any) -> Any:
    if faiss is None:
        raise RuntimeError("未安装 faiss-cpu 或 faiss")
    import numpy as np

    if vectors.size == 0:
        return None
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def _persist_index(
    memory_json: Path,
    index: Any,
    rows: list[dict[str, str]],
    dim: int,
    json_mtime: float,
) -> None:
    ip, mp = semantic_index_paths(memory_json)
    memory_json.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(ip))  # type: ignore[union-attr]
    meta = {"version": 1, "dim": dim, "rows": rows, "json_mtime": json_mtime, "model": (os.getenv("LOCAL_EMBEDDING_MODEL") or _DEFAULT_MODEL).strip()}
    with mp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _load_from_disk(memory_json: Path) -> dict[str, Any] | None:
    if faiss is None:
        return None
    ip, mp = semantic_index_paths(memory_json)
    if not ip.is_file() or not mp.is_file():
        return None
    try:
        with mp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rows = meta.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    try:
        index = faiss.read_index(str(ip))  # type: ignore[union-attr]
    except Exception:
        return None
    return {"index": index, "rows": rows, "dim": int(meta.get("dim", 0)), "json_mtime": float(meta.get("json_mtime", 0))}


def _bundle_fresh_on_disk(memory_json: Path, bundle: dict[str, Any] | None) -> bool:
    """当前 smart_memory.json 的 mtime 未晚于建索引时记录的 mtime，则磁盘索引仍有效。"""
    if not bundle:
        return False
    cur = _json_mtime(memory_json)
    built_at = float(bundle.get("json_mtime", 0))
    return cur <= built_at + _JSON_MTIME_EPS


def get_semantic_bundle(memory_json: Path, store: dict[str, Any]) -> dict[str, Any] | None:
    """
    返回 { index, rows, dim, model_name }；不可用时返回 None。
    使用内存缓存；json 更新后自动重建。
    """
    if not semantic_deps_available() or not _using_faiss_semantic():
        return None

    key = str(memory_json.resolve())
    jmt = _json_mtime(memory_json)
    cached = _bundle_cache.get(key)
    if cached and cached[0] == jmt:
        return cached[1]

    bundle = _load_from_disk(memory_json)
    if _bundle_fresh_on_disk(memory_json, bundle):
        _bundle_cache[key] = (jmt, bundle)
        return bundle

    rows, texts = _rows_and_texts_from_store(store)
    if not rows:
        _bundle_cache[key] = (jmt, None)
        return None

    try:
        model = _load_st_model()
        vecs = _encode_batch(model, texts)
    except Exception:
        _bundle_cache[key] = (jmt, None)
        return None

    import numpy as np

    if vecs.size == 0:
        _bundle_cache[key] = (jmt, None)
        return None

    dim = int(vecs.shape[1])
    index = _build_faiss_index(vecs)
    if index is None:
        _bundle_cache[key] = (jmt, None)
        return None

    try:
        _persist_index(memory_json, index, rows, dim, jmt)
    except Exception:
        pass

    out = {"index": index, "rows": rows, "dim": dim, "json_mtime": jmt, "model": (os.getenv("LOCAL_EMBEDDING_MODEL") or _DEFAULT_MODEL).strip()}
    _bundle_cache[key] = (jmt, out)
    return out


def semantic_search(
    memory_json: Path,
    store: dict[str, Any],
    query: str,
    *,
    top_k: int,
) -> list[tuple[str, str, float]]:
    """
    返回 [(kind, id, score), ...]，score 为内积（归一化向量即余弦相似度）。
    query 为空则返回空列表。
    """
    q = (query or "").strip()
    if not q:
        return []

    bundle = get_semantic_bundle(memory_json, store)
    if not bundle:
        return []

    try:
        model = _load_st_model()
        qv = _encode_query(model, q[:_ENCODE_MAX_CHARS])
    except Exception:
        return []

    index = bundle["index"]
    rows: list[dict[str, str]] = bundle["rows"]
    k = min(max(1, top_k), len(rows))
    scores, idxs = index.search(qv, k)  # type: ignore[union-attr]

    out: list[tuple[str, str, float]] = []
    for i in range(idxs.shape[1]):
        row_i = int(idxs[0, i])
        if row_i < 0 or row_i >= len(rows):
            continue
        sc = float(scores[0, i])
        r = rows[row_i]
        kind = str(r.get("kind", ""))
        rid = str(r.get("id", ""))
        if kind and rid:
            out.append((kind, rid, sc))
    return out


def semantic_index_stats(memory_json: Path, store: dict[str, Any]) -> dict[str, Any]:
    """供 smart_memory_manage stats 展示（不加载 SentenceTransformer，避免首次 stats 即拉模型）。"""
    _ = store
    base: dict[str, Any] = {
        "enabled": bool(_using_faiss_semantic() and semantic_deps_available()),
        "deps": {"faiss": _HAS_FAISS, "sentence_transformers": _HAS_ST},
    }
    ip, mp = semantic_index_paths(memory_json)
    base["index_path"] = str(ip)
    if not base["enabled"]:
        return base
    if not mp.is_file():
        base["rows"] = 0
        base["ready"] = False
        base["index_stale"] = True
        return base
    try:
        with mp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        base["rows"] = 0
        base["ready"] = False
        base["index_stale"] = True
        return base
    rows = meta.get("rows")
    n = len(rows) if isinstance(rows, list) else 0
    base["rows"] = n
    base["dim"] = meta.get("dim")
    base["model"] = meta.get("model")
    base["ready"] = n > 0 and ip.is_file()
    base["index_stale"] = not _bundle_fresh_on_disk(
        memory_json, {"json_mtime": float(meta.get("json_mtime", 0))}
    )
    return base
