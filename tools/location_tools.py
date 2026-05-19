from __future__ import annotations

import os
from typing import Any

from ._util import dumps, err, load_profile


def tool_nearby_restaurants(params: dict[str, Any]) -> str:
    use_mock = bool(params.get("use_mock", False))
    try:
        lat = float(params["lat"])
        lng = float(params["lng"])
    except (KeyError, TypeError, ValueError):
        return err("需要 lat、lng 数字")

    radius = params.get("radius_m", 1500)
    try:
        radius_f = float(radius)
    except (TypeError, ValueError):
        return err("radius_m 须为数字")

    keyword = str(params.get("keyword", "")).strip()
    prof = load_profile()
    taboos_note = prof.get("taboos") or []

    if not use_mock:
        key = os.environ.get("AMAP_KEY") or os.environ.get("GAODE_KEY")
        if not key:
            return (
                "未配置地图 API。请设置环境变量 AMAP_KEY（高德 Web 服务 key），"
                "并在实现中接入周边 POI；或传 use_mock=true 查看演示输出。\n"
                f"收到参数: lat={lat}, lng={lng}, radius_m={radius_f}, keyword={keyword or '无'}\n"
                f"用户禁忌档案（供筛选）: {taboos_note}"
            )

    mock = [
        {
            "name": "演示餐厅·轻食沙拉",
            "distance_m": 420,
            "cuisine": "轻食",
            "rating": 4.6,
            "price_level": "人均约60",
            "tip": "点餐前用 check_taboos 核对配料",
        },
        {
            "name": "演示餐厅·粤菜小馆",
            "distance_m": 890,
            "cuisine": "粤菜",
            "rating": 4.4,
            "price_level": "人均约80",
            "tip": "海鲜类注意过敏与忌口",
        },
    ]
    if keyword:
        mock = [m for m in mock if keyword in m["cuisine"] or keyword in m["name"]] or mock

    return (
        "【演示数据】附近餐厅候选（非真实检索结果）：\n"
        + dumps(mock)
        + "\n禁忌提醒: "
        + (dumps(taboos_note) if taboos_note else "档案未设置禁忌")
    )
