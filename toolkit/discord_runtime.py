"""
Runtime bridge for Discord-aware cheatcode handlers.

Vendor plugins should not import the Discord bot module directly.
Instead, Busy38DiscordBot sets the active bot instance here so
async cheatcode handlers can access Discord APIs.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

_bot: Optional[Any] = None
_controller: Optional[Any] = None


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
