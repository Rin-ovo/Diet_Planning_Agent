from __future__ import annotations

import argparse
import json
import sys
from typing import Annotated, Any

from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.utils.runnable import RunnableCallable
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from llm import ContextBuilder, ContextBuilderConfig, LLMClient
from llm.lc_llm_client_chat import LLMClientChatModel
from storage.user_context import get_current_user_display, set_current_user_id
from tools import append_tool_failure_guidance, run_tool
from tools.runner import REGISTRY, tool_descriptions


def _format_llm_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return (
            "API 认证失败（401）：当前 OPENAI_API_KEY 对该服务无效。\n"
            "请逐项核对：\n"
            "1) 密钥是否从「与 OPENAI_BASE_URL 同一厂商」的控制台复制。\n"
            "2) .env 或 export 的值是否完整、无多余引号/空格/换行。\n"
            "3) 密钥是否已过期、被重置或额度/权限不足。\n"
            f"服务端返回: {exc}"
        )
    if isinstance(exc, PermissionDeniedError):
        return f"API 权限被拒绝：{exc}"
    if isinstance(exc, RateLimitError):
        return f"触发限流或配额不足：{exc}"
    if isinstance(exc, APIConnectionError):
        return f"无法连接模型服务（网络或 base_url 错误）：{exc}"
    if isinstance(exc, BadRequestError):
        low = str(exc).lower()
        if "model" in low or "not exist" in low:
            return (
                "请求被拒（400）：模型在该服务商不存在或未开通（常见：DeepSeek 上不能使用 gpt-4o-mini）。\n"
                "处理：在 .env 设置 OPENAI_MODEL=deepseek-chat（或厂商文档中的模型 id）；"
                "若 base_url 已含 deepseek 且未设 OPENAI_MODEL，LLMClient 会自动使用 deepseek-chat。\n"
                "也可：python chat.py --model deepseek-chat\n"
                f"详情: {exc}"
            )
        return f"请求参数错误（400）：{exc}"
    return f"运行失败：{exc}"


SYSTEM_PROMPT = """你是饮食规划助手。仅通过接口提供的「函数调用」（tool_calls）使用工具，用自然中文回复用户。
你会收到此前多轮对话的完整消息（含 user/assistant/tool），请结合上下文理解用户长期偏好与刚才说过的话。
多用户：系统有「我(me)」「张三(zhangsan)」「李四(lisi)」三套隔离画像与记忆。用户未说明为谁规划时，按当前会话已选用户（默认 me=我）服务；若用户明确要为张三/李四规划，先调用 list_diet_users 确认可选 id，再调用 select_diet_user 切换后再更新档案与推荐。
规则：
- 向用户推荐具体食物、外卖、零食、餐厅名之前：先 get_recent_recommendations 或 filter_recent_recommendations（默认近 3 天）避免重复推荐；再 check_taboos。
- 当你已确定本轮要告诉用户的具体推荐名称时，在最终回复用户之前调用 record_recommended_items 写入这些名称。
- 用户在对话里陈述可记入长期画像的事实时，须主动调用 update_user_profile（不必等用户说「更新档案」）。例如：今天体重/身高/年龄/性别、活动量、忌口、过敏、喜好、目标（减脂等）。体重写入 patch.weight_kg（数字）；身高 height_cm；年龄 age；性别 sex；日常活动量 activity_level（可与 estimate_daily_calories 的 activity 对应）；列表类用 favorites/taboos 等。
- 需要档案或心情记录时用 get_user_profile、log_mood；安慰型零食用 suggest_comfort_foods。
- 热量与配餐用 estimate_daily_calories、lookup_food_calories、suggest_meal_plan；需要一周安排用 suggest_weekly_meal_plan（会先读禁忌，生成后写入档案）。
- 用户表示「周计划里某道菜/某类食材不喜欢」时，调用 add_plan_feedback_exclusions：具体菜名放 dislike_items，不想再见的食材或类别词放 dislike_keywords（如讨厌茄子则写「茄子」），以后配餐与 check_taboos 会避开。
- 菜谱用 search_recipes、estimate_recipe_cost_time；附近餐厅用 nearby_restaurants（无地图 key 时传 use_mock）。
- 智能记忆（闭环）：用户表达稳定偏好、反感、目标变化、对推荐的明确评价时，用 smart_memory_create 写入要点（可加 tags 如 preference/dislike/goal/feedback）；在给出重要推荐前可用 smart_memory_retrieve 拉取相关记忆辅助个性化；片段过多时用 smart_memory_summarize 生成 consolidated 摘要并可归档源片段；重复或分散的条目用 smart_memory_merge；过期或不再适用用 smart_memory_forget（prune_expired、decay、delete 等）；定期 smart_memory_manage 查看 stats 或 vacuum 控容量。
- 可选 Neo4j：默认只读（NEO4J_READ_ONLY=1）：check_taboos / get_user_profile 仅读取图中忌口作参考，不向图写入、不建约束；本地忌口仍以 update_user_profile 写 JSON 为准。若自建库需自动 MERGE，可设 NEO4J_READ_ONLY=0（不建议生产只读库）。
防幻觉与工具失败：
- 系统消息中的档案/智能记忆摘要仅为自动注入的提示，**不得**将其当作已与工具核对的事实；禁忌、热量、具体能否吃某物等**必须以对应工具返回为准**。
- 若任一条 role=tool 的内容以「[错误]」开头、或含「执行异常」「无法解析」等失败描述：**禁止**假装该工具已成功，**禁止**据此编造禁忌结论、热量数字或推荐清单；应说明工具未成功、可改正参数重试，或请用户简化问题/稍后重试。
- 在工具失败或未调用 check_taboos 前，**不要**向用户输出「已确认可吃/不能吃」的确定性结论；可给一般性饮食原则并明确「未做禁忌校验」。
得到工具结果后，用自然中文给用户最终回答；无需再调用工具时直接回复用户。"""


# 同一 graph 实例内，同一工具名连续返回失败时的熔断次数（防死循环刷工具）
_TOOL_FAIL_STREAK_BEFORE_STOP = 3


def _diet_langchain_tools(tool_fail_streak: dict[str, int]) -> list[Any]:
    """LangChain @tool + ToolNode：InjectedState 读当前 messages，InjectedToolCallId 对齐 diet_arg_errors。"""

    tools: list[Any] = []

    for tool_name in sorted(REGISTRY):

        def _make(name: str):
            def _impl(
                msgs: Annotated[dict, InjectedState()],
                tool_call_id: Annotated[str, InjectedToolCallId()],
                **kwargs: Any,
            ) -> str:
                last_msg = msgs["messages"][-1]
                errs: dict[str, str] = {}
                if isinstance(last_msg, AIMessage):
                    errs = (last_msg.additional_kwargs or {}).get("diet_arg_errors") or {}
                if tool_call_id in errs:
                    content: str = append_tool_failure_guidance(errs[tool_call_id])
                else:
                    content = run_tool(name, kwargs)
                fail = isinstance(content, str) and (
                    content.startswith("[错误]")
                    or "执行异常" in content
                    or "无法解析" in content
                    or "不是合法 JSON" in content
                )
                if fail:
                    tool_fail_streak[name] = tool_fail_streak.get(name, 0) + 1
                    if tool_fail_streak[name] >= _TOOL_FAIL_STREAK_BEFORE_STOP:
                        content = (
                            f"{content}\n\n【系统】工具「{name}」已连续失败 {_TOOL_FAIL_STREAK_BEFORE_STOP} 次，"
                            "请停止用相同参数重复调用；向用户说明原因，或换用其它工具/请用户调整需求。"
                        )
                else:
                    tool_fail_streak[name] = 0
                return content

            _impl.__name__ = name
            _impl.__doc__ = tool_descriptions.get(name, name)
            return tool(_impl)

        tools.append(_make(tool_name))
    return tools


def build_graph(client: LLMClient, model: str | None = None) -> Any:
    tool_fail_streak: dict[str, int] = {}
    lc_tools = _diet_langchain_tools(tool_fail_streak)
    bound_model = LLMClientChatModel(
        client=client,
        model_override=model,
        temperature=0.2,
    ).bind_tools(lc_tools)

    def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        response = bound_model.invoke(state["messages"], config)
        if not isinstance(response, AIMessage):
            return {"messages": [AIMessage(content=str(response))]}
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", RunnableCallable(call_model, None))
    workflow.add_node("tools", ToolNode(lc_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def invoke_graph(
    client: LLMClient,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    recursion_limit: int = 40,
) -> tuple[str, list[dict[str, Any]]]:
    lc_messages = convert_to_messages(messages)
    graph = build_graph(client, model)
    final = graph.invoke(
        {"messages": lc_messages},
        config={"recursion_limit": recursion_limit},
    )
    lc_msgs: list[Any] = list(final.get("messages") or [])
    msgs = convert_to_openai_messages(lc_msgs)
    last_assistant = ""
    for m in reversed(lc_msgs):
        if isinstance(m, AIMessage) and not m.tool_calls:
            c = m.content
            if isinstance(c, str):
                last_assistant = c.strip()
                if last_assistant:
                    break
    if not last_assistant:
        for m in reversed(lc_msgs):
            if isinstance(m, AIMessage):
                c = m.content
                if isinstance(c, str) and c.strip():
                    last_assistant = c.strip()
                    break
    return last_assistant, msgs


def run_once(
    client: LLMClient,
    user_text: str,
    *,
    prior_messages: list[dict[str, Any]] | None = None,
    model: str | None = None,
    recursion_limit: int = 40,
    context_builder: ContextBuilder | None = None,
    diet_user_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if diet_user_id:
        set_current_user_id(diet_user_id)
    if prior_messages is None:
        if context_builder is not None:
            seed = context_builder.messages_for_user_turn(
                context_builder.initial_messages(),
                user_text,
            )
        else:
            seed = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
    else:
        seed = list(prior_messages)
        if context_builder is not None and seed and seed[0].get("role") == "system":
            context_builder.refresh_system_message(seed, digest_query=user_text)
        seed.append({"role": "user", "content": user_text})
    return invoke_graph(client, seed, model=model, recursion_limit=recursion_limit)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="饮食规划助手 — LangGraph 编排 + tool_calls，底层使用 LLMClient",
    )
    parser.add_argument("-q", "--query", default="", help="单次提问")
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖默认模型 id（否则使用 LLMClient 推断）",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=40,
        help="LangGraph 最大步数（防止死循环）",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="打印最终完整 messages（调试）",
    )
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=80,
        help="交互模式下截断时保留的最大消息条数（含 tool；不含 system）",
    )
    parser.add_argument(
        "--no-context-digest",
        action="store_true",
        help="关闭 ContextBuilder 自动注入的档案/智能记忆摘要（仍可用工具查询）",
    )
    parser.add_argument(
        "--user",
        default="me",
        metavar="ID",
        help="当前饮食用户：me(我)、zhangsan(张三)、lisi(李四)；也可用中文别名启动后等价",
    )
    args = parser.parse_args()

    try:
        set_current_user_id(args.user)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    try:
        client = LLMClient()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    cb_cfg = ContextBuilderConfig(max_non_system_messages=args.max_history_messages)
    if args.no_context_digest:
        cb_cfg.inject_profile_digest = False
        cb_cfg.inject_smart_memory_digest = False
    context_builder = ContextBuilder(SYSTEM_PROMPT, cb_cfg)
    session: list[dict[str, Any]] = context_builder.initial_messages()

    def one(user_text: str) -> None:
        nonlocal session
        try:
            answer, msgs = invoke_graph(
                client,
                context_builder.messages_for_user_turn(session, user_text),
                model=args.model,
                recursion_limit=args.recursion_limit,
            )
        except Exception as e:
            print(_format_llm_error(e), file=sys.stderr)
            return
        session = msgs
        context_builder.trim_messages(session)
        if args.show_messages:
            print(json.dumps(msgs, ensure_ascii=False, indent=2))
        print("\n=== 助手 ===\n")
        print(answer or "(无文本回复，可加 --show-messages 查看对话与工具结果)")

    if args.query.strip():
        session = context_builder.messages_for_user_turn(
            context_builder.initial_messages(),
            args.query.strip(),
        )
        try:
            answer, msgs = invoke_graph(
                client,
                session,
                model=args.model,
                recursion_limit=args.recursion_limit,
            )
        except Exception as e:
            print(_format_llm_error(e), file=sys.stderr)
            sys.exit(1)
        if args.show_messages:
            print(json.dumps(msgs, ensure_ascii=False, indent=2))
        print("\n=== 助手 ===\n")
        print(answer or "(无文本回复，可加 --show-messages 查看对话与工具结果)")
        return

    print(
        f"当前用户：{get_current_user_display()}（--user 可改）。LangGraph + tool_calls（LLMClient）；"
        "ContextBuilder 每轮刷新档案/智能记忆摘要；可加 --no-context-digest 关闭注入。空行或 quit 退出。",
    )
    while True:
        try:
            line = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "q", "exit"):
            break
        one(line)


if __name__ == "__main__":
    main()
