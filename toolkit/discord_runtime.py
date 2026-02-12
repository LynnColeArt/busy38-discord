"""
Runtime bridge for Discord-aware cheatcode handlers.

Vendor plugins should not import the Discord bot module directly.
Instead, Busy38DiscordBot sets the active bot instance here so
async cheatcode handlers can access Discord APIs.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Optional, Any, Dict

_bot: Optional[Any] = None
_controller: Optional[Any] = None
_active_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("busy38_discord_active_context", default=None)


def set_bot(bot: Any) -> None:
    global _bot
    _bot = bot


def get_bot() -> Any:
    return _bot


def set_controller(controller: Any) -> None:
    global _controller
    _controller = controller


def get_controller() -> Any:
    return _controller


def get_active_context() -> Optional[Dict[str, Any]]:
    return _active_context.get()


@contextlib.contextmanager
def bind_active_context(ctx: Dict[str, Any]):
    token = _active_context.set(ctx)
    try:
        yield
    finally:
        _active_context.reset(token)


async def run_auto_clear_cycle(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute one auto-clear cycle through the active Discord controller.
    """
    ctrl = get_controller()
    if ctrl is None:
        return {"success": False, "skipped": "discord_controller_not_ready"}
    fn = getattr(ctrl, "run_auto_clear_cycle", None)
    if not callable(fn):
        return {"success": False, "skipped": "discord_controller_missing_auto_clear"}
    result = fn(trigger="heartbeat", payload=payload or {})
    if hasattr(result, "__await__"):
        return await result
    return result
