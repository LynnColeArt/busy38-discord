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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import os
import logging
import json
import threading
import time
import uuid

from core.cheatcodes.registry import register_namespace
from .discord_transcript import DiscordTranscriptLogger
from .discord_runtime import (
    get_bot,
    get_controller,
    get_active_context,
    run_auto_clear_cycle,
    run_context_summary_cycle,
)
from .discord_attachments import build_discord_files, close_discord_files, normalize_attachment_specs

logger = logging.getLogger(__name__)
_heartbeat_hook_registered = False
_status_hook_registered = False


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
        if _truthy(os.getenv("DISCORD_AUTO_CLEAR_ENABLE", "0")):
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

        if _truthy(os.getenv("DISCORD_CONTEXT_SUMMARY_ENABLE", "0")):
            summary_interval = max(
                60,
                int(os.getenv("DISCORD_CONTEXT_SUMMARY_INTERVAL_SEC", "3600")),
            )
            manager.register_job(
                name="discord_context_summary",
                interval_seconds=summary_interval,
                source="plugin:busy-38-discord",
                run_immediately=False,
                callback=run_context_summary_cycle,
                metadata={
                    "window_hours": int(os.getenv("DISCORD_CONTEXT_SUMMARY_WINDOW_HOURS", "24")),
                    "min_gap_sec": int(os.getenv("DISCORD_CONTEXT_SUMMARY_MIN_GAP_SEC", "3600")),
                },
            )


def _schedule_coro(coro) -> None:
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except Exception:
        return


def _status_activity_for_cheatcode(namespace: str, action: str, attributes: Dict[str, Any]) -> Optional[str]:
    ns = str(namespace or "").strip().lower()
    act = str(action or "").strip().lower()

    if ns == "rw4":
        if act == "read_file":
            return "opening a file"
        if act == "read_range":
            return "reading a file section"
        if act == "write_file":
            return "writing a file"
        if act == "list":
            return "checking the workspace"
        if act == "shell":
            return "running a command"
        if act == "git_status":
            return "checking git status"
        if act == "git_diff":
            return "reviewing changes"
        if act == "git_commit":
            return "committing changes"
        if act == "lsp_diagnostics":
            return "checking diagnostics"

    if ns == "dlog":
        if act == "search":
            return "searching the channel history"
        if act == "around":
            return "reviewing recent context"

    if ns == "dforum":
        if act == "create_post":
            return "creating a forum post"
        if act == "reply":
            return "posting an update"
        if act == "rename":
            return "updating a thread title"
        if act == "set_tags":
            return "updating thread tags"
        if act == "archive":
            return "archiving a thread"
        if act == "lock":
            return "locking a thread"
        if act == "get_thread":
            return "reviewing a thread"

    if ns == "drelay":
        if act == "emit":
            return "broadcasting an agent status update"
        if act == "read":
            return "checking relay room messages"

    return None


def _status_activity_for_mission(event: str, payload: Dict[str, Any]) -> Optional[str]:
    if event == "mission.started":
        objective = str(payload.get("objective") or "").strip()
        return f"starting mission: {objective}" if objective else "starting mission"
    if event == "mission.step.started":
        step = payload.get("step")
        return f"working on step {step}" if step else "working on a mission step"
    if event == "mission.awaiting_orchestrator":
        step = payload.get("step")
        return f"waiting for guidance for step {step}" if step else "waiting for orchestrator guidance"
    if event == "mission.orchestrator_guidance":
        return "got guidance, continuing execution"
    if event == "mission.step.completed":
        step = payload.get("step")
        return f"completed step {step}" if step else "completed mission step"
    if event == "mission.qa.started":
        return "starting quality check"
    if event == "mission.qa.review":
        attempt = payload.get("attempt")
        state = "passed" if payload.get("approved") else "checking for fixes"
        return f"QA review #{attempt} {state}" if attempt else f"QA review {state}"
    if event == "mission.qa.revision_required":
        return "applying revision requested by QA"
    if event == "mission.qa.revision_started":
        step = payload.get("step")
        return f"working on revision step {step}" if step else "working on revision"
    if event == "mission.qa.revision_completed":
        return "revision complete"
    if event == "mission.approved":
        return "mission complete"
    if event == "mission.failed":
        return "mission failed"
    if event == "mission.cancel_requested":
        return "mission cancelled"
    return None


def _maybe_register_status_hooks() -> None:
    """
    Register hook handlers that can narrate tool/cheatcode progress in Discord.

    Implemented as best-effort. If Busy38 core doesn't provide the hook points,
    this becomes a no-op.
    """
    global _status_hook_registered
    if _status_hook_registered:
        return
    _status_hook_registered = True

    try:
        from core.hooks import on_orchestration_status, on_pre_cheatcode_execute
    except Exception:
        logger.debug("Cheatcode hooks unavailable; skipping discord status hook registration")
        return

    @on_orchestration_status(priority=35)
    def _discord_status_on_mission(run, event: str, payload: Dict[str, Any], context=None):
        ctrl = get_controller()
        if ctrl is None:
            return
        ctx = get_active_context() or {}
        ch = ctx.get("channel")
        if ch is None:
            return

        activity = _status_activity_for_mission(event, payload or {})
        if not activity:
            return

        terminal_events = {
            "mission.approved",
            "mission.failed",
            "mission.cancel_requested",
        }
        if event in terminal_events:
            clear_fn = getattr(ctrl, "_status_clear", None)
            if callable(clear_fn):
                _schedule_coro(clear_fn(ch))
            return

        _schedule_coro(ctrl.post_status(ch, activity))

    @on_pre_cheatcode_execute(priority=40)
    def _discord_status_on_cheatcode(namespace: str, action: str, attributes: Dict[str, Any], context=None):
        ctrl = get_controller()
        if ctrl is None:
            return
        ctx = get_active_context() or {}
        ch = ctx.get("channel")
        if ch is None:
            return

        activity = _status_activity_for_cheatcode(namespace, action, attributes or {})
        if not activity:
            return

        # Keep it lightweight: narration is controlled by the controller env flags.
        _schedule_coro(ctrl.post_status(ch, activity))


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


@dataclass
class _DiscordRelayStore:
    data_dir: str = "./data/memory"
    room_max_events: int = 200
    global_max_events: int = 5000

    def __post_init__(self) -> None:
        relay_dir = Path(self.data_dir)
        relay_dir.mkdir(parents=True, exist_ok=True)
        self._path = relay_dir / "discord_agent_relay.jsonl"
        self._lock = threading.Lock()
        self._room_max_events = max(10, int(os.getenv("DISCORD_RELAY_ROOM_MAX_EVENTS", str(self.room_max_events))))
        self._global_max_events = max(self._room_max_events, int(os.getenv("DISCORD_RELAY_GLOBAL_MAX_EVENTS", str(self.global_max_events))))

    def _iter_events(self) -> Iterable[Dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fp:
            for raw in fp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    logger.warning("Skipping malformed relay bus line in %s", self._path)
                    continue
                if isinstance(event, dict):
                    yield event

    def _load_events(self) -> Dict[str, list]:
        events_by_room: Dict[str, list] = {}
        for event in self._iter_events():
            room_id = str(event.get("room_id") or "").strip()
            if not room_id:
                continue
            events_by_room.setdefault(room_id, []).append(event)
        return events_by_room

    def _sort_and_prune(self, events_by_room: Dict[str, list]) -> Dict[str, list]:
        by_room: Dict[str, list] = {}
        all_events = []
        for room_id, rows in events_by_room.items():
            rows.sort(key=lambda item: int(item.get("ts_epoch", 0)))
            keep_rows = rows[-self._room_max_events :]
            by_room[room_id] = keep_rows
            all_events.extend(keep_rows)

        if len(all_events) <= self._global_max_events:
            return by_room

        all_events.sort(key=lambda item: int(item.get("ts_epoch", 0)))
        all_events = all_events[-self._global_max_events :]
        keep_global_ids = {str(item.get("event_id")) for item in all_events}
        for room_id, rows in by_room.items():
            by_room[room_id] = [r for r in rows if str(r.get("event_id")) in keep_global_ids]
        return by_room

    def _write_events(self, events_by_room: Dict[str, list]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            for rows in events_by_room.values():
                for event in rows:
                    fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        tmp.replace(self._path)

    def append_event(self, *, room_id: str, agent_id: str, kind: str, message: str, visibility: str, metadata: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None, correlation_id: Optional[str] = None, actor: Optional[str] = None) -> Dict[str, Any]:
        room = room_id.strip()
        if not room:
            return {"success": False, "error": "room_id is required"}

        event = {
            "event_id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "ts_epoch": int(time.time()),
            "room_id": room,
            "agent_id": str(agent_id or "unknown").strip()[:80],
            "kind": str(kind or "status").strip()[:40],
            "message": str(message or "").strip(),
            "visibility": str(visibility or "public").strip().lower(),
            "metadata": dict(metadata or {}),
            "run_id": run_id,
            "correlation_id": correlation_id,
            "actor": str(actor or "discord_plugin").strip(),
        }

        with self._lock:
            events_by_room = self._load_events()
            events_by_room.setdefault(room, []).append(event)
            events_by_room = self._sort_and_prune(events_by_room)
            self._write_events(events_by_room)

        return {"success": True, "event_id": event["event_id"], "room_id": room, "visibility": event["visibility"]}

    def read_events(self, *, room_id: str, visibility: str = "public", kinds: Optional[str] = None, limit: int = 20, since_event_id: Optional[str] = None) -> Dict[str, Any]:
        requested_vis = str(visibility or "public").strip().lower()
        kind_filter = set(part.strip().lower() for part in str(kinds or "").split(",") if part.strip())

        events_by_room = self._sort_and_prune(self._load_events())
        rows = events_by_room.get(room_id.strip(), [])
        out: list = []
        seen = False

        for event in rows:
            event_visibility = str(event.get("visibility", "public"))
            if event_visibility != requested_vis and requested_vis != "any":
                continue
            if kind_filter:
                if str(event.get("kind", "")).strip().lower() not in kind_filter:
                    continue
            if since_event_id and not seen:
                if str(event.get("event_id")) == str(since_event_id):
                    seen = True
                continue
            out.append(event)

        out.sort(key=lambda item: int(item.get("ts_epoch", 0)), reverse=True)
        limit = max(1, min(200, int(limit)))
        out = out[:limit]
        return {"success": True, "room_id": room_id, "visibility": requested_vis, "count": len(out), "events": out}


class _DiscordRelayHandler:
    def __init__(self):
        self._store = _DiscordRelayStore(data_dir=os.getenv("BUSY38_CHATLOG_DIR", "./data/memory"))

    def execute(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        action_map = {
            "emit": self._emit,
            "read": self._read,
            "status": self._status,
        }
        fn = action_map.get(action)
        if not fn:
            return {"success": False, "error": f"Unknown action: {action}"}
        return fn(**kwargs)

    def _normalize_room(self, room_id: str) -> str:
        return str(room_id or "").strip()

    def _emit(
        self,
        room_id: str,
        agent_id: str,
        message: str,
        kind: str = "status",
        visibility: str = "public",
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> Dict[str, Any]:
        room = self._normalize_room(room_id)
        if not room:
            return {"success": False, "error": "room_id is required"}
        clean_kind = str(kind or "status").strip().lower()
        clean_visibility = str(visibility or "public").strip().lower()
        if clean_visibility not in {"public", "ops", "private", "any"}:
            return {"success": False, "error": f"invalid visibility: {visibility}"}
        if len(str(message or "")) > 3000:
            return {"success": False, "error": "message too long (max 3000)"}
        return self._store.append_event(
            room_id=room,
            agent_id=agent_id,
            kind=clean_kind,
            message=str(message or ""),
            visibility=clean_visibility if clean_visibility != "any" else "public",
            metadata=metadata,
            run_id=run_id,
            correlation_id=correlation_id,
            actor=actor,
        )

    def _read(
        self,
        room_id: str,
        visibility: str = "public",
        kinds: Optional[str] = None,
        limit: int = 20,
        since_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        room = self._normalize_room(room_id)
        if not room:
            return {"success": False, "error": "room_id is required"}
        if int(limit) <= 0:
            return {"success": False, "error": "limit must be > 0"}
        return self._store.read_events(
            room_id=room,
            visibility=visibility,
            kinds=kinds,
            limit=int(limit),
            since_event_id=since_event_id,
        )

    def _status(self, room_id: Optional[str] = None) -> Dict[str, Any]:
        room = self._normalize_room(room_id or "all")
        all_events = self._store._load_events()
        if room == "all":
            count = sum(len(events) for events in all_events.values())
            rooms = {key: len(events) for key, events in all_events.items()}
            return {"success": True, "room": "all", "rooms": rooms, "count": count}
        return {
            "success": True,
            "room": room,
            "count": len(all_events.get(room, [])),
        }


class Toolkit:
    """Vendor plugin entry point (auto-instantiated by PluginManager)."""

    def __init__(self):
        data_dir = os.getenv("BUSY38_CHATLOG_DIR", "./data/memory")
        register_namespace("dlog", _DiscordLogHandler(data_dir=data_dir))
        register_namespace("dforum", _DiscordForumHandler())
        register_namespace("drelay", _DiscordRelayHandler())
        _maybe_register_heartbeat_jobs()
        _maybe_register_status_hooks()


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
