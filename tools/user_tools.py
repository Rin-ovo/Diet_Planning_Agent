from __future__ import annotations

from typing import Any

from storage.user_context import (
    DISPLAY_NAME,
    get_current_user_display,
    get_current_user_id,
    normalize_user_id,
    set_current_user_id,
)

from ._util import dumps, err


def tool_select_diet_user(params: dict[str, Any]) -> str:
    """
    切换当前服务的用户画像（我 / 张三 / 李四）。
    参数 user_id 可为 me、zhangsan、lisi 或 我、张三、李四。
    """
    raw = params.get("user_id", params.get("user", ""))
    if raw is None or str(raw).strip() == "":
        return err("需要 user_id（如 me、zhangsan、lisi 或 我、张三、李四）")
    try:
        uid = set_current_user_id(str(raw).strip())
    except ValueError as e:
        return err(str(e))
    disp = DISPLAY_NAME.get(uid, uid)
    return dumps(
        {
            "ok": True,
            "当前用户_id": uid,
            "显示名": disp,
            "提示": "后续画像、推荐记录、智能记忆均针对该用户；未说明为谁规划时 CLI 默认 me（我）。",
        }
    )


def tool_list_diet_users(params: dict[str, Any]) -> str:
    _ = params
    cur = get_current_user_id()
    rows = []
    for uid in ("me", "zhangsan", "lisi"):
        disp = DISPLAY_NAME[uid]
        rows.append(
            {
                "user_id": uid,
                "显示名": disp,
                "是否当前": uid == cur,
            }
        )
    return dumps(
        {
            "当前用户": get_current_user_display(),
            "当前用户_id": cur,
            "可选用户": rows,
        }
    )
