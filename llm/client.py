from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from openai import AsyncOpenAI, OpenAI


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class LLMClient:
    """
    调用任意兼容 OpenAI Chat Completions HTTP 接口的服务。
    默认使用流式响应；可通过环境变量 OPENAI_API_KEY、OPENAI_BASE_URL 配置。
    可选 OPENAI_MODEL（或 LLM_DEFAULT_MODEL）指定默认模型 id，须与服务商一致。
    若 base_url 指向 DeepSeek 且未指定模型，则默认使用 deepseek-chat。
    初始化时会尝试加载项目根目录下的 .env（若已安装 python-dotenv）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        _try_load_dotenv()
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        key = (key or "").strip()
        url = (url or "").strip() or None
        if not key:
            raise ValueError(
                "未配置 API Key：请设置环境变量 OPENAI_API_KEY，"
                "或在 LLMClient(api_key='...') 中传入。"
                "也可在项目根目录创建 .env 并写入 OPENAI_API_KEY=你的密钥"
                "（使用 DeepSeek 等兼容服务时同时设置 OPENAI_BASE_URL）。"
            )
        url_l = (url or "").lower()
        env_model = (
            os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_DEFAULT_MODEL") or ""
        ).strip()
        if default_model and str(default_model).strip():
            self._default_model = str(default_model).strip()
        elif env_model:
            self._default_model = env_model
        elif "deepseek" in url_l:
            self._default_model = "deepseek-chat"
        else:
            self._default_model = "gpt-4o-mini"

        client_kw: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if url:
            client_kw["base_url"] = url
        self._client = OpenAI(**client_kw)
        self._async_client = AsyncOpenAI(**client_kw)

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def astream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        stream = await self._async_client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def complete_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            stream=False,
            **kwargs,
        )
        if not resp.choices:
            return ""
        msg = resp.choices[0].message
        return (msg.content or "") if msg else ""

    @staticmethod
    def _assistant_message_to_dict(msg: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"role": getattr(msg, "role", "assistant")}
        content = getattr(msg, "content", None)
        out["content"] = content if content else None
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            serialized: list[dict[str, Any]] = []
            for tc in tcs:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                serialized.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(fn, "name", ""),
                            "arguments": getattr(fn, "arguments", "") or "{}",
                        },
                    }
                )
            out["tool_calls"] = serialized
        return out

    def complete_chat_turn(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        非流式一轮对话，支持 OpenAI 格式 tools / tool_calls（供 LangGraph 工作流调用）。
        返回可直接追加到 messages 的 assistant 字典。
        """
        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            tools=list(tools),
            tool_choice=tool_choice,
            temperature=temperature,
            **kwargs,
        )
        if not resp.choices:
            return {"role": "assistant", "content": ""}
        return self._assistant_message_to_dict(resp.choices[0].message)

    async def acomplete_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        resp = await self._async_client.chat.completions.create(
            model=model or self._default_model,
            messages=list(messages),
            stream=False,
            **kwargs,
        )
        if not resp.choices:
            return ""
        msg = resp.choices[0].message
        return (msg.content or "") if msg else ""
