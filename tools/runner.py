from __future__ import annotations

from typing import Any, Callable

# OpenAI Chat Completions「tools」项：与 LLMClient.complete_chat_turn / LangChain StructuredTool 对齐。
OPENAI_TOOL_PARAMETERS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
    "description": "该工具的参数字典，与 run_tool 第二参数一致（JSON 对象）。",
}

from .comfort_tools import tool_suggest_comfort_foods
from .location_tools import tool_nearby_restaurants
from .memory_tools import (
    tool_filter_recent_recommendations,
    tool_get_recent_recommendations,
    tool_record_recommended_items,
)
from .nutrition_tools import (
    tool_estimate_daily_calories,
    tool_lookup_food_calories,
    tool_suggest_meal_plan,
    tool_suggest_weekly_meal_plan,
)
from .plan_tools import tool_add_plan_feedback_exclusions
from .profile_tools import (
    tool_check_taboos,
    tool_get_user_profile,
    tool_log_mood,
    tool_update_user_profile,
)
from .recipe_tools import tool_estimate_recipe_cost_time, tool_search_recipes
from .smart_memory_tools import (
    tool_smart_memory_create,
    tool_smart_memory_forget,
    tool_smart_memory_manage,
    tool_smart_memory_merge,
    tool_smart_memory_retrieve,
    tool_smart_memory_summarize,
)
from .user_tools import tool_list_diet_users, tool_select_diet_user

ToolFn = Callable[[dict[str, Any]], str]

REGISTRY: dict[str, ToolFn] = {
    "get_recent_recommendations": tool_get_recent_recommendations,
    "filter_recent_recommendations": tool_filter_recent_recommendations,
    "record_recommended_items": tool_record_recommended_items,
    "get_user_profile": tool_get_user_profile,
    "update_user_profile": tool_update_user_profile,
    "check_taboos": tool_check_taboos,
    "log_mood": tool_log_mood,
    "suggest_comfort_foods": tool_suggest_comfort_foods,
    "estimate_daily_calories": tool_estimate_daily_calories,
    "lookup_food_calories": tool_lookup_food_calories,
    "suggest_meal_plan": tool_suggest_meal_plan,
    "suggest_weekly_meal_plan": tool_suggest_weekly_meal_plan,
    "add_plan_feedback_exclusions": tool_add_plan_feedback_exclusions,
    "search_recipes": tool_search_recipes,
    "estimate_recipe_cost_time": tool_estimate_recipe_cost_time,
    "nearby_restaurants": tool_nearby_restaurants,
    "smart_memory_create": tool_smart_memory_create,
    "smart_memory_retrieve": tool_smart_memory_retrieve,
    "smart_memory_summarize": tool_smart_memory_summarize,
    "smart_memory_forget": tool_smart_memory_forget,
    "smart_memory_merge": tool_smart_memory_merge,
    "smart_memory_manage": tool_smart_memory_manage,
    "select_diet_user": tool_select_diet_user,
    "list_diet_users": tool_list_diet_users,
}

tool_descriptions: dict[str, str] = {
    "get_recent_recommendations": "查询近 N 天已推荐给用户的饮食名称。params: {within_days?: 默认3}",
    "filter_recent_recommendations": "从候选列表中排除近 N 天已推荐项。params: {candidates: [], within_days?: 3}",
    "record_recommended_items": "将本轮最终推荐给用户的饮食名称写入档案（用于去重）。params: {items: []}",
    "get_user_profile": "读取饮食档案（喜好、忌口、心情等）；若启用 Neo4j 会附带图库状态与图中忌口词。params: {}",
    "update_user_profile": "根据对话中用户陈述更新档案。patch 可含 weight_kg/height_cm/age/sex/activity_level、favorites、taboos、allergies、goals、preferred_cuisines、notes、comfort_snacks_ok 等；写 weight_kg 会自动记 weight_updated_at。params: {patch: dict, list_op?: append|set}",
    "check_taboos": "校验候选食物是否触犯禁忌/过敏。params: {items: [], extra_forbidden?: []}",
    "log_mood": "记录心情。params: {mood: str, mood_note?: str}",
    "suggest_comfort_foods": "心情不好时的奶茶/零食等建议（会避开禁忌）。params: {mood?, force?}",
    "estimate_daily_calories": "估算 TDEE 与目标摄入。params: {sex, age, height_cm, weight_kg, activity, goal, weekly_kg_change?}",
    "lookup_food_calories": "查询常见食物参考热量。params: {items: []}",
    "suggest_meal_plan": "按目标热量分餐（演示）。params: {daily_target_kcal, num_meals?}",
    "suggest_weekly_meal_plan": "生成 7 天饮食框架并写入档案 weekly_meal_plan；尊重禁忌与档案。params: {daily_target_kcal, num_meals_per_day?}",
    "add_plan_feedback_exclusions": "用户对计划中的食物反感时：写入禁忌。params: {dislike_items?:[], dislike_keywords?:[], reason_note?}",
    "search_recipes": "按时间/预算/关键词筛选菜谱。params: {max_time_min?, max_budget_cny?, keywords?}",
    "estimate_recipe_cost_time": "估算菜谱耗时与成本。params: {recipe_id|recipe_name} 或 {ingredients, prep_min?, cook_min?}",
    "nearby_restaurants": "附近餐厅（演示或接地图 API）。params: {lat, lng, radius_m?, keyword?, use_mock?}",
    "smart_memory_create": "创建一条对话记忆（偏好/结论），供后续检索与推荐调权。params: {text, tags?:[], source?: dialogue, salience?:0.05~2, ttl_days?:可选过期}",
    "smart_memory_retrieve": "检索记忆：关键词/标签过滤；若已安装 faiss-cpu+sentence-transformers 且 query 非空，则合并 BGE-M3+FAISS 语义召回（use_faiss 默认 true，可用 SMART_MEMORY_FAISS=0 关闭）。params: {query?:, tags_any?:[], limit?:12, use_faiss?:true, include_archived?:false, include_consolidated?:true}",
    "smart_memory_summarize": "将多条 episode 规则拼接为一条 consolidated 摘要（可选归档源）。params: {tag?:, max_episodes?:10, archive_sources?:true, title?:}",
    "smart_memory_forget": "遗忘/清理：delete_episode|delete_consolidated|archive_episode|decay_episode|decay_consolidated|prune_expired|prune_low_salience。params: {mode, id?, factor?, min_salience?}",
    "smart_memory_merge": "整合多条 episode 为一条新记忆并可归档来源。params: {source_ids:[], merged_text?:, tags?:[], archive_sources?:true, salience?:}",
    "smart_memory_manage": "管理闭环：stats|list_recent|vacuum（容量整理）。params: {action, k?}",
    "select_diet_user": "切换当前饮食规划所服务的用户（画像/推荐记录/智能记忆按用户隔离）。params: {user_id: me|zhangsan|lisi 或 我|张三|李四}",
    "list_diet_users": "列出可选用户及当前会话用户。params: {}",
}


def openai_tool_specs() -> list[dict[str, Any]]:
    """供 LLM `tools` 参数的 OpenAI 风格 function 定义列表（名称与 REGISTRY 一致）。"""
    specs: list[dict[str, Any]] = []
    for name in sorted(REGISTRY):
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_descriptions.get(name, name),
                    "parameters": dict(OPENAI_TOOL_PARAMETERS_JSON_SCHEMA),
                },
            }
        )
    return specs


_TOOL_FAILURE_REMINDER = (
    "\n【系统提醒】工具未成功执行时：不得将禁忌校验、热量数值、具体推荐菜名等伪造为已通过本工具确认；"
    "请向用户简短说明失败或不确定原因，改正参数后重试，或建议稍后重试。"
)


def list_tools() -> str:
    lines = [f"- {n}: {tool_descriptions.get(n, '')}" for n in sorted(REGISTRY)]
    return "可用工具:\n" + "\n".join(lines)


def _with_tool_failure_reminder(msg: str) -> str:
    if _TOOL_FAILURE_REMINDER.strip() in msg:
        return msg
    return msg + _TOOL_FAILURE_REMINDER


def append_tool_failure_guidance(msg: str) -> str:
    """给模型看的工具失败类文案追加防幻觉提醒（供 graph 层 JSON 校验等使用）。"""
    return _with_tool_failure_reminder(msg)


def run_tool(name: str, params: dict[str, Any] | None = None) -> str:
    if name not in REGISTRY:
        return _with_tool_failure_reminder(
            f"[错误] 未知工具: {name}。可用: {', '.join(sorted(REGISTRY))}"
        )
    p = params if isinstance(params, dict) else {}
    try:
        out = REGISTRY[name](p)
    except Exception as e:
        return _with_tool_failure_reminder(f"[错误] {name} 执行异常: {e}")
    if isinstance(out, str) and out.startswith("[错误]"):
        return _with_tool_failure_reminder(out)
    return out
