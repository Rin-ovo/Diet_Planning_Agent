"""
记忆存储门面：按用户解析智能记忆文件路径。
向量/图谱检索可在本类中组合 Neo4jStoreStub、smart_memory_faiss（BGE-M3+FAISS）等再输出给 Agent。
"""
from __future__ import annotations

from pathlib import Path

from .document_store import DocumentStore
from .user_context import get_current_user_id


def smart_memory_file_for_current_user() -> Path:
    return DocumentStore(get_current_user_id()).smart_memory_path()
