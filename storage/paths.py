"""数据目录根路径（供 storage 与 tools 共用，避免循环依赖）。"""
from __future__ import annotations

from pathlib import Path


def project_data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d
