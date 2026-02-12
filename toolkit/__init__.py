#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""
busy-38-discord vendor plugin toolkit.

Registers agent-internal cheatcodes for querying Discord transcripts:
- dlog:search (broad search with snippets, recency-biased)
- dlog:around (drill-down around a message id)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
import os
import logging

from core.cheatcodes.registry import register_namespace
from .discord_transcript import DiscordTranscriptLogger
from .discord_runtime import get_bot, run_auto_clear_cycle
from .discord_attachments import build_discord_files, close_discord_files, normalize_attachment_specs

logger = logging.getLogger(__name__)
_heartbeat_hook_registered = False


def _truthy(raw: str) -> bool:
    v = (raw or "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _maybe_register_heartbeat_jobs() -> None:
    """
    Register heartbeat hook callback that installs discord auto-clear jobs.

    This remains plugin-local and only activates if heartbeat hooks are available.
    """
    global _heartbeat_hook_registered
    if _heartbeat_hook_registered:
        return
    _heartbeat_hook_registered = True

    try:
        from core.hooks import on_heartbeat_register_jobs
    except Exception:
        logger.debug("Heartbeat hooks unavailable; skipping discord auto-clear hook registration")
        return

    @on_heartbeat_register_jobs(priority=20)
    def _register_discord_jobs(manager, context=None):
        if not _truthy(os.getenv("DISCORD_AUTO_CLEAR_ENABLE", "0")):
            return
        interval = max(60, int(os.getenv("DISCORD_AUTO_CLEAR_INTERVAL_SEC", "900")))
        manager.register_job(
            name="discord_auto_clear",
            interval_seconds=interval,
            source="plugin:busy-38-discord",
            run_immediately=False,
            callback=run_auto_clear_cycle,
            metadata={
                "window_hours": int(os.getenv("DISCORD_CLEAR_WINDOW_HOURS", "72")),
                "min_gap_sec": int(os.getenv("DISCORD_AUTO_CLEAR_MIN_GAP_SEC", "21600")),
            },
        )


@dataclass
class _DiscordLogHandler:
    data_dir: str = "./data/memory"

    def __post_init__(self) -> None:
        self._logger = DiscordTranscriptLogger(data_dir=self.data_dir)

    def execute(self, action: str, **kwargs: Any) -> Any:
        action_map = {
            "search": self._search,
            "around": self._around,
        }
        fn = action_map.get(action)
        if not fn:
            return {"success": False, "error": f"Unknown action: {action}"}
        return fn(**kwargs)

    def _search(
        self,
        query: str,
        project_id: Optional[str] = None,
        max_age_hours: int = 24,
        max_messages: int = 5000,
        context: int = 80,
        case_sensitive: bool = False,
        regex: bool = False,
        snippets_per_message: int = 2,
        max_results: int = 20,
        **_: Any,
    ) -> Dict[str, Any]:
        since = None
        if max_age_hours and int(max_age_hours) > 0:
            since = datetime.now(timezone.utc) - timedelta(hours=int(max_age_hours))
        results = self._logger.search(
            query=query,
            project_id=project_id,
            since=since,
            max_messages=int(max_messages),
            context=int(context),
            case_sensitive=bool(case_sensitive),
            regex=bool(regex),
            snippets_per_message=int(snippets_per_message),
            max_results=int(max_results),
        )
        return {"success": True, "results": results}

    def _around(
        self,
        message_id: str,
        before: int = 8,
        after: int = 8,
        **_: Any,
    ) -> Dict[str, Any]:
        mid = str(message_id).strip()
        if mid.isdigit():
            source_id = f"discord:{mid}"
        else:
            source_id = mid
        rows = self._logger.context_around(source_id=source_id, before=int(before), after=int(after))
        return {"success": True, "rows": rows}


class Toolkit:
    """Vendor plugin entry point (auto-instantiated by PluginManager)."""

    def __init__(self):
        data_dir = os.getenv("BUSY38_CHATLOG_DIR", "./data/memory")
        register_namespace("dlog", _DiscordLogHandler(data_dir=data_dir))
        register_namespace("dforum", _DiscordForumHandler())
        _maybe_register_heartbeat_jobs()


class _DiscordForumHandler:
    """
    Async Discord forum/thread operations.

    Intended for agent use when interacting with a forum-based task board.
    """

    async def execute(self, action: str, **kwargs: Any) -> Any:
        action_map = {
            "reply": self._reply,
            "rename": self._rename,
            "set_tags": self._set_tags,
            "archive": self._archive,
            "lock": self._lock,
            "create_post": self._create_post,
            "get_thread": self._get_thread,
        }
        fn = action_map.get(action)
        if not fn:
            return {"success": False, "error": f"Unknown action: {action}"}
        return await fn(**kwargs)

    async def _fetch_channel(self, channel_id: int):
        bot = get_bot()
        if bot is None:
            raise RuntimeError("Discord runtime not initialized (no bot instance).")
        ch = bot.get_channel(channel_id)
        if ch is None:
            ch = await bot.fetch_channel(channel_id)
        return ch

    async def _reply(self, thread_id: int, content: str, attachments: Any = None, **_: Any) -> Dict[str, Any]:
        ch = await self._fetch_channel(int(thread_id))
        files, attachment_errors = await build_discord_files(normalize_attachment_specs(attachments))
        try:
            msg = await ch.send(content=str(content or ""), files=files or None)
        finally:
            close_discord_files(files)
        out = {"success": True, "message_id": msg.id, "attachments_sent": len(files)}
        if attachment_errors:
            out["attachment_errors"] = attachment_errors
        return out

    async def _rename(self, thread_id: int, name: str, **_: Any) -> Dict[str, Any]:
        th = await self._fetch_channel(int(thread_id))
        await th.edit(name=str(name))
        return {"success": True}

    async def _archive(self, thread_id: int, archived: bool = True, **_: Any) -> Dict[str, Any]:
        th = await self._fetch_channel(int(thread_id))
        await th.edit(archived=bool(archived))
        return {"success": True}

    async def _lock(self, thread_id: int, locked: bool = True, **_: Any) -> Dict[str, Any]:
        th = await self._fetch_channel(int(thread_id))
        await th.edit(locked=bool(locked))
        return {"success": True}

    async def _set_tags(self, thread_id: int, tag_names: str = "", **_: Any) -> Dict[str, Any]:
        bot = get_bot()
        if bot is None:
            raise RuntimeError("Discord runtime not initialized (no bot instance).")
        th = await self._fetch_channel(int(thread_id))
        parent = getattr(th, "parent", None)
        if parent is None or not hasattr(parent, "available_tags"):
            return {"success": False, "error": "Thread has no forum parent / tags not supported"}

        wanted = [t.strip() for t in str(tag_names).split(",") if t.strip()]
        tags = []
        for wt in wanted:
            for tag in parent.available_tags:
                if tag.name.lower() == wt.lower():
                    tags.append(tag)
                    break

        await th.edit(applied_tags=tags)
        return {"success": True, "applied": [t.name for t in tags]}

    async def _create_post(
        self,
        forum_id: int,
        title: str,
        content: str,
        tag_names: str = "",
        attachments: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        forum = await self._fetch_channel(int(forum_id))
        wanted = [t.strip() for t in str(tag_names).split(",") if t.strip()]
        tags = []
        if hasattr(forum, "available_tags"):
            for wt in wanted:
                for tag in forum.available_tags:
                    if tag.name.lower() == wt.lower():
                        tags.append(tag)
                        break

        thread = await forum.create_thread(name=str(title), content=str(content), applied_tags=tags)
        # discord.py returns (thread, message) sometimes depending on version; normalize.
        th = thread[0] if isinstance(thread, (tuple, list)) else thread
        out: Dict[str, Any] = {"success": True, "thread_id": th.id}
        files, attachment_errors = await build_discord_files(normalize_attachment_specs(attachments))
        if files:
            try:
                attachment_msg = await th.send(content="Attachments", files=files or None)
                out["attachment_message_id"] = attachment_msg.id
                out["attachments_sent"] = len(files)
            finally:
                close_discord_files(files)
        if attachment_errors:
            out["attachment_errors"] = attachment_errors
        return out

    async def _get_thread(self, thread_id: int, limit: int = 30, **_: Any) -> Dict[str, Any]:
        th = await self._fetch_channel(int(thread_id))
        # Fetch recent messages
        msgs = []
        async for m in th.history(limit=int(limit), oldest_first=False):
            msgs.append(
                {
                    "id": m.id,
                    "author": str(m.author),
                    "author_id": m.author.id,
                    "content": m.content,
                    "created_at": m.created_at.isoformat() if getattr(m, "created_at", None) else None,
                    "attachments": [
                        {
                            "id": int(getattr(a, "id", 0) or 0),
                            "filename": str(getattr(a, "filename", "") or "attachment"),
                            "size": int(getattr(a, "size", 0) or 0),
                            "content_type": getattr(a, "content_type", None),
                            "url": getattr(a, "url", None),
                        }
                        for a in (getattr(m, "attachments", None) or [])
                    ],
                }
            )
        msgs.reverse()
        applied = [t.name for t in getattr(th, "applied_tags", []) or []]
        return {"success": True, "thread_id": th.id, "name": getattr(th, "name", ""), "applied_tags": applied, "messages": msgs}
