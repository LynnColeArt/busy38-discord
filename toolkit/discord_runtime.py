"""
Runtime bridge for Discord-aware cheatcode handlers.

Vendor plugins should not import the Discord bot module directly.
Instead, Busy38DiscordBot sets the active bot instance here so
async cheatcode handlers can access Discord APIs.
"""

from __future__ import annotations

from typing import Optional, Any

_bot: Optional[Any] = None


def set_bot(bot: Any) -> None:
    global _bot
    _bot = bot


def get_bot() -> Any:
    return _bot

