from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ._util import dumps, err, load_profile, save_profile

FOOD_KCAL: dict[str, tuple[float, str]] = {
    "米饭": (116, "100g"),
    "糙米": (111, "100g"),
    "燕麦": (389, "100g"),
    "鸡胸肉": (165, "100g"),
    "鸡蛋": (144, "100g"),
    "牛奶": (54, "100ml"),
    "酸奶": (72, "100g"),
    "西蓝花": (34, "100g"),
    "番茄": (18, "100g"),
    "黄瓜": (16, "100g"),
    "苹果": (52, "100g"),
    "香蕉": (89, "100g"),
    "奶茶": (350, "一杯"),
    "薯片": (536, "100g"),
    "黑巧克力": (598, "100g"),
    "豆腐": (76, "100g"),
    "三文鱼": (208, "100g"),
    "瘦牛肉": (250, "100g"),
    "全麦面包": (246, "100g"),
}

MEAL_TEMPLATES: list[dict[str, Any]] = [
    {"name": "早餐燕麦酸奶水果", "tags": ["燕麦", "酸奶", "苹果"], "cuisine": "轻食"},
    {"name": "午餐鸡胸糙米饭蔬菜", "tags": ["鸡胸肉", "糙米", "西蓝花"], "cuisine": "中式"},
    {"name": "晚餐清蒸鱼配蔬菜", "tags": ["三文鱼", "番茄", "黄瓜"], "cuisine": "中式"},
    {"name": "加餐坚果牛奶", "tags": ["牛奶"], "cuisine": "通用"},
    {"name": "午餐瘦牛肉全麦三明治", "tags": ["瘦牛肉", "全麦面包", "番茄"], "cuisine": "简餐"},
    {"name": "晚餐豆腐蔬菜杂粮饭", "tags": ["豆腐", "糙米", "西蓝花"], "cuisine": "中式"},
    {"name": "早餐鸡蛋牛奶香蕉", "tags": ["鸡蛋", "牛奶", "香蕉"], "cuisine": "通用"},
]

_WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _taboo_set(prof: dict[str, Any]) -> set[str]:
    return {str(x) for x in (prof.get("taboos") or []) + (prof.get("allergies") or [])}


def _template_blocked(t: dict[str, Any], taboos: set[str]) -> bool:
    blob = t["name"] + "".join(t["tags"])
    return any(b and b in blob for b in taboos)


def _usable_templates(prof: dict[str, Any]) -> list[dict[str, Any]]:
    taboos = _taboo_set(prof)
    usable = [t for t in MEAL_TEMPLATES if not _template_blocked(t, taboos)]
    return usable if usable else list(MEAL_TEMPLATES)


def _norm_key(s: str) -> str:
    return str(s).strip().lower()


def tool_lookup_food_calories(params: dict[str, Any]) -> str:
    items = params.get("items")
    if not isinstance(items, list) or not items:
        return err("items 须为非空列表")
    rows: list[dict[str, Any]] = []
    for raw in items:
        name = str(raw).strip()
        key = next((k for k in FOOD_KCAL if _norm_key(k) == _norm_key(name)), None)
        if key is None:
            key = next((k for k in FOOD_KCAL if k in name or name in k), None)
        if key:
            kcal, note = FOOD_KCAL[key]
            rows.append({"查询名": name, "参考条目": key, "千卡每100g或说明": kcal, "备注": note})
        else:
            rows.append({"查询名": name, "参考条目": None, "说明": "库中无精确条目，请查证营养成分表"})
    return dumps(rows)


def tool_estimate_daily_calories(params: dict[str, Any]) -> str:
    try:
        age = float(params["age"])
        h = float(params["height_cm"])
        w = float(params["weight_kg"])
    except (KeyError, TypeError, ValueError):
        return err("需要数值字段 age, height_cm, weight_kg")

    sex = str(params.get("sex", "")).strip().lower()
    if sex in ("男", "m", "male"):
        bmr = 10 * w + 6.25 * h - 5 * age + 5
    elif sex in ("女", "f", "female"):
        bmr = 10 * w + 6.25 * h - 5 * age - 161
    else:
        return err('sex 须为 male/female 或 男/女')

    act = str(params.get("activity", "sedentary")).strip().lower()
    mult_map = {
        "sedentary": 1.2,
        "久坐": 1.2,
        "light": 1.375,
        "轻度": 1.375,
        "moderate": 1.55,
        "中度": 1.55,
        "active": 1.725,
        "高度": 1.725,
    }
    mult = mult_map.get(act, 1.2)

    tdee = bmr * mult
    goal = str(params.get("goal", "maintain")).strip().lower()
    try:
        weekly_f = float(params.get("weekly_kg_change", 0.3))
    except (TypeError, ValueError):
        weekly_f = 0.3

    delta = 0.0
    if goal in ("lose", "减脂", "减重"):
        delta = -7700 * weekly_f / 7
    elif goal in ("gain", "增重", "增肌"):
        delta = 7700 * min(weekly_f, 0.5) / 7

    target = max(800.0, tdee + delta)
    out = {
        "BMR_估算": round(bmr, 0),
        "活动系数": mult,
        "TDEE_估算": round(tdee, 0),
        "目标摄入_估算": round(target, 0),
        "说明": "仅为通用估算，疾病/孕期/运动员等请咨询医生或营养师。",
    }
    return dumps(out)


def tool_suggest_meal_plan(params: dict[str, Any]) -> str:
    try:
        total = float(params["daily_target_kcal"])
    except (KeyError, TypeError, ValueError):
        return err("需要 daily_target_kcal 数字")
    n = int(params.get("num_meals", 3) or 3)
    if n < 2 or n > 6:
        return err("num_meals 建议在 2~6")

    prof = load_profile()
    cuisines = [str(x) for x in (prof.get("preferred_cuisines") or [])]
    usable = _usable_templates(prof)

    per = total / n
    plan: list[dict[str, Any]] = []
    for i in range(n):
        t = usable[i % len(usable)]
        kcal = round(per * (0.9 if i == 0 else 1.05 if i == n - 1 else 1.0))
        plan.append(
            {
                "餐次": i + 1,
                "建议框架": t["name"],
                "参考菜系": t["cuisine"],
                "本餐目标热量约": kcal,
                "提示": "具体食材请再用 check_taboos 与 lookup_food_calories 核对。",
            }
        )

    hint = f"偏好菜系档案: {cuisines or '未设置'}"
    return hint + "\n" + dumps(plan)


def tool_suggest_weekly_meal_plan(params: dict[str, Any]) -> str:
    try:
        total = float(params["daily_target_kcal"])
    except (KeyError, TypeError, ValueError):
        return err("需要 daily_target_kcal（可先 estimate_daily_calories 再传入）")
    n = int(params.get("num_meals_per_day", 3) or 3)
    if n < 2 or n > 5:
        return err("num_meals_per_day 建议 2~5")

    prof = load_profile()
    usable = _usable_templates(prof)
    cuisines = prof.get("preferred_cuisines") or []

    days_out: list[dict[str, Any]] = []
    idx = 0
    for d in range(7):
        label = _WEEKDAY_LABELS[d]
        per = total / n
        meals: list[dict[str, Any]] = []
        for m in range(n):
            t = usable[idx % len(usable)]
            idx += 1
            kcal = round(per * (0.9 if m == 0 else 1.05 if m == n - 1 else 1.0))
            meals.append(
                {
                    "餐次": m + 1,
                    "建议框架": t["name"],
                    "参考菜系": t["cuisine"],
                    "本餐目标热量约": kcal,
                }
            )
        days_out.append(
            {
                "周几": label,
                "weekday_index": d + 1,
                "日目标热量约": round(total),
                "餐次": meals,
            }
        )

    prof["weekly_meal_plan"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "daily_target_kcal": total,
        "num_meals_per_day": n,
        "days": days_out,
    }
    save_profile(prof)

    hint = (
        f"偏好菜系: {cuisines or '未设置'}。\n"
        "若用户不喜欢计划中某道菜或某类食材，请调用 add_plan_feedback_exclusions："
        "dislike_items 写具体菜名，dislike_keywords 写食材/类别，以后模板与推荐会避开。\n"
    )
    return hint + dumps({"week_plan": days_out})
