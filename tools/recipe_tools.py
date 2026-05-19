from __future__ import annotations

from typing import Any

from ._util import dumps, err

RECIPES: list[dict[str, Any]] = [
    {
        "id": "r1",
        "name": "蒜蓉西蓝花鸡胸肉",
        "cuisine": "中式",
        "prep_min": 10,
        "cook_min": 15,
        "cost_cny_estimate": 25,
        "steps": "鸡胸肉切丁腌制，西蓝花焯水，少油快炒，蒜末增香。",
        "tags": ["低脂", "高蛋白"],
    },
    {
        "id": "r2",
        "name": "番茄鸡蛋全麦三明治",
        "cuisine": "简餐",
        "prep_min": 8,
        "cook_min": 12,
        "cost_cny_estimate": 15,
        "steps": "番茄炒蛋少油，夹全麦面包，可配生菜。",
        "tags": ["快手"],
    },
    {
        "id": "r3",
        "name": "三文鱼蔬菜碗",
        "cuisine": "轻食",
        "prep_min": 15,
        "cook_min": 10,
        "cost_cny_estimate": 45,
        "steps": "三文鱼煎或烤，配糙米饭与焯水蔬菜。",
        "tags": ["优质脂肪"],
    },
]


def tool_search_recipes(params: dict[str, Any]) -> str:
    max_t = params.get("max_time_min")
    max_b = params.get("max_budget_cny")
    kw = params.get("keywords")
    kws: list[str] = []
    if isinstance(kw, str) and kw.strip():
        kws = [kw.strip()]
    elif isinstance(kw, list):
        kws = [str(x).strip() for x in kw if str(x).strip()]

    out: list[dict[str, Any]] = []
    for r in RECIPES:
        total_time = int(r["prep_min"]) + int(r["cook_min"])
        if max_t is not None:
            try:
                if total_time > float(max_t):
                    continue
            except (TypeError, ValueError):
                return err("max_time_min 须为数字")
        if max_b is not None:
            try:
                if float(r["cost_cny_estimate"]) > float(max_b):
                    continue
            except (TypeError, ValueError):
                return err("max_budget_cny 须为数字")
        if kws:
            blob = r["name"] + "".join(r.get("tags", []))
            if not any(k in blob for k in kws):
                continue
        out.append(r)
    return dumps(out) if out else "无匹配菜谱。"


def tool_estimate_recipe_cost_time(params: dict[str, Any]) -> str:
    rid = params.get("recipe_id")
    rname = params.get("recipe_name")
    if rid or rname:
        for r in RECIPES:
            if rid and r["id"] == str(rid):
                return dumps(
                    {
                        "菜谱": r["name"],
                        "准备分钟": r["prep_min"],
                        "烹饪分钟": r["cook_min"],
                        "合计分钟": int(r["prep_min"]) + int(r["cook_min"]),
                        "成本估算_元": r["cost_cny_estimate"],
                    }
                )
            if rname and str(rname) in r["name"]:
                return dumps(
                    {
                        "菜谱": r["name"],
                        "准备分钟": r["prep_min"],
                        "烹饪分钟": r["cook_min"],
                        "合计分钟": int(r["prep_min"]) + int(r["cook_min"]),
                        "成本估算_元": r["cost_cny_estimate"],
                    }
                )
        return err("未找到菜谱")

    ingredients = params.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        return err("需要 recipe_id/recipe_name 或 ingredients")
    try:
        prep = int(params.get("prep_min", 10))
        cook = int(params.get("cook_min", 15))
    except (TypeError, ValueError):
        return err("prep_min/cook_min 须为整数")

    return dumps(
        {
            "食材数": len(ingredients),
            "粗算成本_元": 5 * len(ingredients),
            "准备分钟": prep,
            "烹饪分钟": cook,
            "合计分钟": prep + cook,
        }
    )
