from __future__ import annotations

import importlib.util
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str):
    path = ROOT / "toolkit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


discord_state = _load_module("discord_state")
ChannelKey = discord_state.ChannelKey
DiscordStateStore = discord_state.DiscordStateStore
MessageRecord = discord_state.MessageRecord


def _msg(author_id: int, is_bot: bool, content: str, *, created_at: float | None = None) -> MessageRecord:
    return MessageRecord(
        message_id=author_id,
        created_at_unix=created_at if created_at is not None else time.time(),
        author_id=author_id,
        author_name=f"user-{author_id}",
        content=content,
        is_bot=is_bot,
        attachments=[{"filename": "a.txt", "size": 42}],
    )


def test_channel_key_formatting():
    assert ChannelKey(None, 10).as_str() == "dm:10"
    assert ChannelKey(99, 10).as_str() == "99:10"


def test_ingest_tracks_participants_and_bot_participants():
    store = DiscordStateStore()
    key = ChannelKey(1, 2)
    store.ingest_message(key, _msg(10, is_bot=False, content="hello"))
    store.ingest_message(key, _msg(11, is_bot=True, content="hi"))

    cfg = store.channel_config(key)
    assert cfg.subscribed is False
    assert cfg.follow_mode is False
    assert cfg.history_limit == 80


def test_prompt_context_omits_bots_when_disabled():
    store = DiscordStateStore()
    key = ChannelKey(1, 2)
    store.ingest_message(key, _msg(10, is_bot=False, content="human"))
    store.ingest_message(key, _msg(11, is_bot=True, content="bot"))

    lines = store.prompt_context_lines(key, include_bots=False, max_messages=10)
    assert len(lines) == 1
    assert lines[0].startswith("user-10 (10): human")
    assert "attachments" in lines[0]


def test_prompt_context_limits_history_and_max_age():
    store = DiscordStateStore(default_history_limit=2)
    key = ChannelKey(1, 2)
    now = time.time()
    store.ingest_message(key, MessageRecord(1, now - 100, 1, "u1", "old", False))
    store.ingest_message(key, MessageRecord(2, now - 1, 2, "u2", "new", False))

    recent = store.prompt_context_lines(key, max_messages=2, max_age_sec=10)
    assert recent == ["u2 (2): new"]


def test_subscription_import_export_roundtrip():
    store = DiscordStateStore()
    key = ChannelKey(123, 456)
    store.set_subscribed(key, True)
    store.set_follow_mode(key, True)
    store.set_history_limit(key, 100)

    exported = store.export_subscriptions()
    assert "123:456" in exported
    assert exported["123:456"]["follow_mode"] is True

    fresh = DiscordStateStore()
    fresh.import_subscriptions(exported)
    assert fresh.channel_config(key).subscribed is True
    assert fresh.channel_config(key).follow_mode is True
    assert fresh.channel_config(key).history_limit == 100
