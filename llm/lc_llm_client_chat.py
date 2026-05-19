"""LangChain BaseChatModel 适配 LLMClient，供 LangGraph ReAct 预置范式调用。"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence, cast

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field

from tools.runner import openai_tool_specs

from .client import LLMClient


def assistant_openai_dict_to_ai_message(d: dict[str, Any]) -> AIMessage:
    """将 LLMClient.complete_chat_turn 返回的 assistant 字典转为 AIMessage（含 diet_arg_errors 侧道信息）。"""
    content = d.get("content") or ""
    diet_arg_errors: dict[str, str] = {}
    lc_tool_calls: list[dict[str, Any]] = []
    for tc in d.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tid = str(tc.get("id") or "")
        name = (fn.get("name") or "").strip()
        raw = fn.get("arguments") or "{}"
        parse_err: str | None = None
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            parse_err = (
                f"[错误] 工具 {name or '(未命名)'} 的 arguments 不是合法 JSON: {e}。"
                "【请勿编造工具执行结果；请用合法 JSON 对象作为 arguments 重新调用。】"
            )
            args = {}
        if parse_err is None and not isinstance(args, dict):
            parse_err = (
                f"[错误] 工具 {name or '(未命名)'} 的 arguments 解析后不是 JSON 对象。"
                "【请勿编造工具执行结果；请传入一个 JSON 对象。】"
            )
        if parse_err is not None:
            diet_arg_errors[tid] = parse_err
        lc_tool_calls.append(
            {
                "name": name,
                "args": args if isinstance(args, dict) else {},
                "id": tid,
                "type": "tool_call",
            }
        )
    add_kw: dict[str, Any] = {}
    if diet_arg_errors:
        add_kw["diet_arg_errors"] = diet_arg_errors
    return AIMessage(content=content, tool_calls=lc_tool_calls, additional_kwargs=add_kw)


class LLMClientChatModel(BaseChatModel):
    """通过 OpenAI 兼容 HTTP 调用底层服务；bind_tools 使用与 REGISTRY 一致的 OpenAI tools 列表。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(description="LLMClient 实例")
    model_override: Optional[str] = Field(default=None, description="覆盖默认模型 id")
    temperature: float = Field(default=0.2)

    @property
    def _llm_type(self) -> str:
        return "llm_client_chat"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        by_name = {spec["function"]["name"]: spec for spec in openai_tool_specs()}
        ordered: list[dict[str, Any]] = []
        for t in tools:
            n = getattr(t, "name", None)
            if not n or n not in by_name:
                raise ValueError(f"bind_tools: 未知工具 {t!r}")
            ordered.append(by_name[n])
        tc = tool_choice if tool_choice is not None else "auto"
        return cast(
            Runnable[Any, Any],
            self.bind(tools=ordered, tool_choice=tc, **kwargs),
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        for k in (
            "ls_structured_output_format",
            "ls_structured_output_format_dict",
            "structured_output_format",
            "tags",
            "metadata",
        ):
            kwargs.pop(k, None)
        tools = kwargs.pop("tools", None) or []
        tool_choice = kwargs.pop("tool_choice", "auto")
        oai = convert_to_openai_messages(messages)
        adict = self.client.complete_chat_turn(
            oai,
            tools=tools,
            model=self.model_override,
            temperature=self.temperature,
            tool_choice=tool_choice,
            **kwargs,
        )
        ai_msg = assistant_openai_dict_to_ai_message(adict)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
