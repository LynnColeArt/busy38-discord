from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_actions():
    actions_spec = importlib.util.spec_from_file_location(
        "busy38_discord_ui_actions",
        ROOT / "ui" / "actions.py",
    )
    assert actions_spec is not None
    actions = importlib.util.module_from_spec(actions_spec)
    assert actions_spec.loader is not None
    actions_spec.loader.exec_module(actions)
    return actions, actions._POLICY_MODULE


def test_evaluate_scope_prefers_specific_deny_then_allow(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    _actions, policy = _load_actions()

    scope = {
        "mode": "custom",
        "rules": [
            {"target": "guild", "ids": ["10"], "action": "allow"},
            {"target": "channel", "ids": ["20"], "action": "allow"},
            {"target": "user", "ids": ["30"], "action": "deny"},
        ],
    }
    normalized, reason_codes, errors = policy.validate_scope(scope)
    assert errors == []
    assert reason_codes == []
    assert normalized is not None

    denied = policy.evaluate_scope(
        normalized,
        feature="commands",
        guild_id=10,
        channel_id=20,
        role_ids=[99],
        user_id=30,
    )
    assert denied["allowed"] is False
    assert denied["reason_code"] == "DISCORD_SCOPE_RULE_DENY"
    assert denied["matched_rule"]["target"] == "user"

    allowed = policy.evaluate_scope(
        normalized,
        feature="commands",
        guild_id=10,
        channel_id=20,
        role_ids=[99],
        user_id=31,
    )
    assert allowed["allowed"] is True
    assert allowed["matched_rule"]["target"] == "channel"


def test_handle_scope_rejects_invalid_payload_with_reason_codes(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()

    result = actions.handle_scope(
        {"mode": "custom", "rules": [{"target": "guild", "ids": ["abc"], "action": "allow"}]},
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )
    assert result["success"] is False
    assert "DISCORD_SCOPE_RULE_IDS_INVALID" in result["reason_codes"]

    audit_lines = (tmp_path / "audit.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert audit_lines
    last_record = json.loads(audit_lines[-1])
    assert last_record["action_id"] == "scope"
    assert last_record["success"] is False
    assert "DISCORD_SCOPE_RULE_IDS_INVALID" in last_record["reason_codes"]


def test_handle_settings_persists_and_debug_reflects_state(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()
    context = {
        "plugin_id": "busy-38-discord",
        "actor": "sam",
        "source_path": str(ROOT),
    }

    settings_result = actions.handle_settings(
        {
            "feature_flags": {
                "status_narration_enabled": True,
                "status_narration_mode": "send",
                "attachments": {
                    "include_urls": False,
                    "text_preview_enabled": True,
                    "text_preview_max_bytes": 2048,
                    "text_preview_max_chars": 240,
                },
            },
            "runtime": {
                "command_prefix": "!busy38-admin ",
                "anti_spam": {
                    "window_sec": 12,
                    "max_events": 4,
                    "cooldown_sec": 30,
                },
            },
        },
        "POST",
        context,
    )
    assert settings_result["success"] is True

    debug_result = actions.handle_debug({}, "GET", context)
    assert debug_result["success"] is True
    policy_payload = debug_result["payload"]["policy"]
    assert policy_payload["feature_flags"]["status_narration_enabled"] is True
    assert policy_payload["feature_flags"]["status_narration_mode"] == "send"
    assert policy_payload["feature_flags"]["attachments"]["include_urls"] is False
    assert policy_payload["runtime"]["command_prefix"] == "!busy38-admin "
    assert policy_payload["runtime"]["anti_spam"]["max_events"] == 4


def test_handle_debug_rejects_non_get(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()

    result = actions.handle_debug(
        {},
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam", "source_path": str(ROOT)},
    )

    assert result["success"] is False
    assert result["reason_codes"] == ["DISCORD_UI_METHOD_INVALID"]

    audit_lines = (tmp_path / "audit.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert audit_lines
    last_record = json.loads(audit_lines[-1])
    assert last_record["action_id"] == "debug"
    assert last_record["success"] is False
    assert last_record["reason_codes"] == ["DISCORD_UI_METHOD_INVALID"]


def test_handle_validate_returns_preview_and_reason_codes(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()

    result = actions.handle_validate(
        {
            "scope": {"mode": "custom", "rules": []},
            "settings": {
                "feature_flags": {"status_narration_mode": "edit"},
                "runtime": {"max_invoke_interval_sec": 1.25},
            },
        },
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )
    assert result["success"] is True
    assert "DISCORD_SCOPE_CUSTOM_EMPTY" in result["reason_codes"]
    assert result["payload"]["policy_preview"]["runtime"]["max_invoke_interval_sec"] == 1.25


def test_handle_settings_rejects_whitespace_only_command_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()

    result = actions.handle_settings(
        {
            "runtime": {
                "command_prefix": "   ",
            },
        },
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )
    assert result["success"] is False
    assert "DISCORD_SETTINGS_VALUE_INVALID" in result["reason_codes"]
    assert "runtime.command_prefix must be a non-empty string" in result["errors"]


def test_handle_settings_rejects_non_boolean_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(tmp_path / "audit.ndjson"))
    actions, _policy = _load_actions()

    result = actions.handle_settings(
        {
            "enabled": "false",
        },
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )

    assert result["success"] is False
    assert "DISCORD_SETTINGS_VALUE_INVALID" in result["reason_codes"]
    assert "enabled must be boolean" in result["errors"]
    assert not (tmp_path / "policy.json").exists()


def test_handle_scope_fails_closed_when_audit_write_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(tmp_path / "policy.json"))
    audit_path = tmp_path / "audit-dir"
    audit_path.mkdir()
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(audit_path))
    actions, _policy = _load_actions()

    result = actions.handle_scope(
        {"mode": "all", "rules": []},
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )

    assert result["success"] is False
    assert "DISCORD_POLICY_AUDIT_FAILED" in result["reason_codes"]
    assert not (tmp_path / "policy.json").exists()


def test_handle_settings_fails_closed_when_audit_write_fails(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "scope": {"mode": "all", "effective_source": "saved", "rules": []},
                "feature_flags": {"status_narration_enabled": False},
                "runtime": {"command_prefix": "!busy38 "},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BUSY38_DISCORD_POLICY_PATH", str(policy_path))
    audit_path = tmp_path / "audit-dir"
    audit_path.mkdir()
    monkeypatch.setenv("BUSY38_DISCORD_UI_AUDIT_PATH", str(audit_path))
    actions, _policy = _load_actions()

    before = json.loads(policy_path.read_text(encoding="utf-8"))
    result = actions.handle_settings(
        {
            "enabled": False,
        },
        "POST",
        {"plugin_id": "busy-38-discord", "actor": "sam"},
    )

    assert result["success"] is False
    assert "DISCORD_POLICY_AUDIT_FAILED" in result["reason_codes"]
    after = json.loads(policy_path.read_text(encoding="utf-8"))
    assert after == before
