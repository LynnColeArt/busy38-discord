"""
Discord state utilities for Busy38.

Pure-python state store:
- Rolling per-channel history (for prompt context)
- Discovered participants and other bot agents
- Subscription + follow-mode configuration
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ChannelKey:
    guild_id: Optional[int]
    channel_id: int

    def as_str(self) -> str:
        g = "dm" if self.guild_id is None else str(self.guild_id)
        return f"{g}:{self.channel_id}"


@dataclass
class MessageRecord:
    message_id: int
    created_at_unix: float
    author_id: int
    author_name: str
    content: str
    is_bot: bool
    reply_to_id: Optional[int] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChannelConfig:
    subscribed: bool = False
    follow_mode: bool = False  # if True, bot may respond more proactively (still rate-limited)
    history_limit: int = 80


@dataclass
class ChannelState:
    cfg: ChannelConfig = field(default_factory=ChannelConfig)
    history: Deque[MessageRecord] = field(default_factory=lambda: deque(maxlen=80))
    participants: Set[int] = field(default_factory=set)  # user IDs seen in channel
    bot_participants: Set[int] = field(default_factory=set)  # bot IDs seen in channel

    def _ensure_maxlen(self) -> None:
        # keep deque maxlen in sync with cfg.history_limit
        if self.history.maxlen != self.cfg.history_limit:
            old = list(self.history)
            self.history = deque(old[-self.cfg.history_limit :], maxlen=self.cfg.history_limit)

    def ingest(self, rec: MessageRecord) -> None:
        self._ensure_maxlen()
        self.history.append(rec)
        self.participants.add(rec.author_id)
        if rec.is_bot:
            self.bot_participants.add(rec.author_id)


class DiscordStateStore:
    def __init__(self, default_history_limit: int = 80):
        self._default_history_limit = default_history_limit
        self._channels: Dict[str, ChannelState] = {}
        self._guild_agents: Dict[int, Set[int]] = defaultdict(set)

    def _get(self, key: ChannelKey) -> ChannelState:
        k = key.as_str()
        if k not in self._channels:
            st = ChannelState()
            st.cfg.history_limit = self._default_history_limit
            st.history = deque(maxlen=st.cfg.history_limit)
            self._channels[k] = st
        return self._channels[k]

    def ingest_message(self, key: ChannelKey, rec: MessageRecord) -> None:
        st = self._get(key)
        st.ingest(rec)
        if key.guild_id is not None and rec.is_bot:
            self._guild_agents[key.guild_id].add(rec.author_id)

    def set_subscribed(self, key: ChannelKey, subscribed: bool) -> None:
        self._get(key).cfg.subscribed = subscribed

    def set_follow_mode(self, key: ChannelKey, follow_mode: bool) -> None:
        self._get(key).cfg.follow_mode = follow_mode

    def set_history_limit(self, key: ChannelKey, limit: int) -> None:
        st = self._get(key)
        st.cfg.history_limit = int(limit)
        st._ensure_maxlen()

    def reset_channel_history(self, key: ChannelKey) -> None:
        """
        Clear rolling history/participants for a single channel while preserving config.
        """
        st = self._get(key)
        st.history.clear()
        st.participants.clear()
        st.bot_participants.clear()

    def channel_config(self, key: ChannelKey) -> ChannelConfig:
        return self._get(key).cfg

    def list_subscriptions(self) -> List[Tuple[ChannelKey, ChannelConfig]]:
        out: List[Tuple[ChannelKey, ChannelConfig]] = []
        for k, st in self._channels.items():
            if st.cfg.subscribed:
                guild, chan = k.split(":", 1)
                out.append(
                    (
                        ChannelKey(guild_id=None if guild == "dm" else int(guild), channel_id=int(chan)),
                        st.cfg,
                    )
                )
        return out

    def known_bot_agents(self, guild_id: Optional[int]) -> Set[int]:
        if guild_id is None:
            return set()
        return set(self._guild_agents.get(guild_id, set()))

    def prompt_context_lines(
        self,
        key: ChannelKey,
        *,
        max_messages: int = 40,
        include_bots: bool = True,
        max_age_sec: Optional[int] = None,
    ) -> List[str]:
        st = self._get(key)
        msgs = list(st.history)
        if max_age_sec is not None:
            now = time.time()
            msgs = [m for m in msgs if (now - float(m.created_at_unix)) <= float(max_age_sec)]
        msgs = msgs[-max_messages:]
        lines: List[str] = []
        for m in msgs:
            if not include_bots and m.is_bot:
                continue
            # Keep formatting stable; Discord IDs help identify agents.
            who = f"{m.author_name} ({m.author_id})"
            line = f"{who}: {m.content}"
            if m.attachments:
                names: List[str] = []
                for att in m.attachments[:4]:
                    filename = str(att.get("filename") or "file")
                    size = att.get("size")
                    if isinstance(size, int):
                        names.append(f"{filename} ({size}B)")
                    else:
                        names.append(filename)
                if len(m.attachments) > 4:
                    names.append(f"+{len(m.attachments) - 4} more")
                line += f" [attachments: {', '.join(names)}]"
            lines.append(line)
        return lines

    def export_subscriptions(self) -> Dict[str, Dict]:
        data: Dict[str, Dict] = {}
        for k, st in self._channels.items():
            if st.cfg.subscribed:
                data[k] = {
                    "subscribed": True,
                    "follow_mode": bool(st.cfg.follow_mode),
                    "history_limit": int(st.cfg.history_limit),
                }
        return data

    def import_subscriptions(self, data: Dict[str, Dict]) -> None:
        for k, cfg in (data or {}).items():
            try:
                guild, chan = k.split(":", 1)
                key = ChannelKey(guild_id=None if guild == "dm" else int(guild), channel_id=int(chan))
                st = self._get(key)
                st.cfg.subscribed = bool(cfg.get("subscribed", True))
                st.cfg.follow_mode = bool(cfg.get("follow_mode", False))
                st.cfg.history_limit = int(cfg.get("history_limit", self._default_history_limit))
                st._ensure_maxlen()
            except Exception:
                # Ignore corrupted entries
                continue


def load_subscriptions_from_keystore(keystore) -> Optional[Dict[str, Dict]]:
    try:
        record = keystore.get_password(
            agent_id="busy38",
            name="discord_subscriptions",
            actor="discord_bot",
        )
        if not record:
            return None
        return json.loads(record.password)
    except Exception:
        return None


def save_subscriptions_to_keystore(keystore, data: Dict[str, Dict], *, actor: str = "discord_bot") -> None:
    keystore.save_password(
        agent_id="busy38",
        name="discord_subscriptions",
        password=json.dumps(data, separators=(",", ":"), sort_keys=True),
        metadata={"updated_at_unix": time.time(), "count": len(data)},
        actor=actor,
    )
