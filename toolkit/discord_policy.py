from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional


# This helper exists so UI handlers and runtime code can share one explicit
# policy schema/validation path without diverging on authority behavior.
_SCOPE_MODES = {"all", "custom", "none"}
_RULE_TARGETS = {"guild", "channel", "role", "user"}
_RULE_ACTIONS = {"allow", "deny"}
_RULE_FEATURES = {"commands", "admin", "auto_clear", "context_summary", "attachments", "status"}
_STATUS_MODES = {"edit", "send"}
_SCOPE_SPECIFICITY = {"guild": 1, "channel": 2, "role": 3, "user": 4}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _truthy_env(name: str, default: str) -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        value = int(default)
    return max(minimum, value)


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = float(default)
    return max(minimum, value)


def _string_env(name: str, default: str) -> str:
    value = str(os.getenv(name, default) or "").strip()
    return value or default


def _string_list_env(name: str, default: str) -> list[str]:
    output: list[str] = []
    for part in str(os.getenv(name, default) or "").split(","):
        item = part.strip()
        if item:
            output.append(item)
    return output


def _plugin_root(context: Optional[Dict[str, Any]] = None) -> Path:
    context_dict = context if isinstance(context, dict) else {}
    source_path = str(context_dict.get("source_path") or "").strip()
    if source_path:
        return Path(source_path).resolve()
    return Path(__file__).resolve().parents[1]


def policy_path(context: Optional[Dict[str, Any]] = None) -> Path:
    context_dict = context if isinstance(context, dict) else {}
    explicit = str(context_dict.get("policy_path") or os.getenv("BUSY38_DISCORD_POLICY_PATH", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (_plugin_root(context_dict) / "data" / "discord_policy.json").resolve()


def audit_path(context: Optional[Dict[str, Any]] = None) -> Path:
    context_dict = context if isinstance(context, dict) else {}
    explicit = str(context_dict.get("audit_path") or os.getenv("BUSY38_DISCORD_UI_AUDIT_PATH", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (_plugin_root(context_dict) / "data" / "discord_ui_actions.ndjson").resolve()


def default_policy() -> Dict[str, Any]:
    return {
        "enabled": True,
        "scope": {
            "mode": "all",
            "effective_source": "env-default",
            "rules": [],
        },
        "feature_flags": {
            "auto_clear_enabled": _truthy_env("DISCORD_AUTO_CLEAR_ENABLE", "0"),
            "auto_clear_min_gap_sec": _int_env("DISCORD_AUTO_CLEAR_MIN_GAP_SEC", 21600, minimum=60),
            "context_summary_enabled": _truthy_env("DISCORD_CONTEXT_SUMMARY_ENABLE", "0"),
            "context_summary_interval_sec": _int_env("DISCORD_CONTEXT_SUMMARY_INTERVAL_SEC", 3600, minimum=60),
            "no_response_reaction_enabled": _truthy_env("DISCORD_NO_RESPONSE_REACTIONS", "1"),
            "no_response_reaction_palette": _string_list_env("DISCORD_NO_RESPONSE_EMOJIS", "👍,👀,✅"),
            "status_narration_enabled": _truthy_env("DISCORD_STATUS_ENABLE", "0"),
            "status_narration_mode": _string_env("DISCORD_STATUS_MODE", "edit").lower(),
            "respond_with_reaction": _truthy_env("DISCORD_NO_RESPONSE_REACTIONS", "1"),
            "allowlist_admin_users_only": False,
            "attachments": {
                "include_urls": _truthy_env("DISCORD_ATTACHMENT_INCLUDE_URLS", "1"),
                "text_preview_enabled": _truthy_env("DISCORD_ATTACHMENT_TEXT_PREVIEW_ENABLE", "1"),
                "text_preview_max_bytes": _int_env("DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_BYTES", 65536, minimum=1),
                "text_preview_max_chars": _int_env("DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_CHARS", 1200, minimum=80),
            },
        },
        "runtime": {
            "command_prefix": "!busy38 ",
            "max_invoke_interval_sec": _float_env("DISCORD_MIN_INVOKE_INTERVAL_SEC", 6.0, minimum=0.1),
            "anti_spam": {
                "window_sec": _int_env("DISCORD_FOLLOW_SPAM_WINDOW_SEC", 30, minimum=1),
                "max_events": _int_env("DISCORD_FOLLOW_SPAM_MAX_EVENTS", 12, minimum=0),
                "cooldown_sec": _int_env("DISCORD_FOLLOW_SPAM_COOLDOWN_SEC", 45, minimum=0),
            },
        },
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _record_hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_id_list(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    normalized: list[str] = []
    for raw in raw_ids:
        candidate = str(raw).strip()
        if not candidate or not candidate.isdigit():
            return []
        normalized.append(candidate)
    return normalized


def _normalize_feature_list(raw_features: Any) -> Optional[list[str]]:
    if raw_features is None:
        return []
    if not isinstance(raw_features, list):
        return None
    normalized: list[str] = []
    for raw in raw_features:
        feature = str(raw).strip().lower()
        if not feature:
            continue
        if feature not in _RULE_FEATURES:
            return None
        if feature not in normalized:
            normalized.append(feature)
    return normalized


def validate_scope(scope: Any) -> tuple[Optional[Dict[str, Any]], list[str], list[str]]:
    if not isinstance(scope, dict):
        return None, ["DISCORD_SCOPE_PAYLOAD_INVALID"], ["scope payload must be an object"]

    reason_codes: list[str] = []
    errors: list[str] = []

    mode = str(scope.get("mode") or "").strip().lower()
    if mode not in _SCOPE_MODES:
        return None, ["DISCORD_SCOPE_MODE_INVALID"], ["scope.mode must be all, custom, or none"]

    raw_rules = scope.get("rules", [])
    if not isinstance(raw_rules, list):
        return None, ["DISCORD_SCOPE_PAYLOAD_INVALID"], ["scope.rules must be an array"]

    normalized_rules: list[Dict[str, Any]] = []
    seen_rules: set[str] = set()
    deny_only = True
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            errors.append(f"scope.rules[{index}] must be an object")
            if "DISCORD_SCOPE_PAYLOAD_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_PAYLOAD_INVALID")
            continue

        target = str(raw_rule.get("target") or "").strip().lower()
        if target not in _RULE_TARGETS:
            errors.append(f"scope.rules[{index}].target is invalid")
            if "DISCORD_SCOPE_RULE_TARGET_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_RULE_TARGET_INVALID")
            continue

        action = str(raw_rule.get("action") or "").strip().lower()
        if action not in _RULE_ACTIONS:
            errors.append(f"scope.rules[{index}].action is invalid")
            if "DISCORD_SCOPE_RULE_ACTION_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_RULE_ACTION_INVALID")
            continue

        ids = _normalize_id_list(raw_rule.get("ids"))
        if not ids:
            errors.append(f"scope.rules[{index}].ids must be a non-empty numeric-id array")
            if "DISCORD_SCOPE_RULE_IDS_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_RULE_IDS_INVALID")
            continue

        features = _normalize_feature_list(raw_rule.get("features"))
        if features is None:
            errors.append(f"scope.rules[{index}].features contains an invalid feature")
            if "DISCORD_SCOPE_RULE_FEATURE_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_RULE_FEATURE_INVALID")
            continue

        normalized_rule = {
            "target": target,
            "ids": ids,
            "action": action,
        }
        if features:
            normalized_rule["features"] = features

        fingerprint = _record_hash(normalized_rule)
        if fingerprint in seen_rules:
            errors.append(f"scope.rules[{index}] duplicates an existing rule")
            if "DISCORD_SCOPE_RULE_DUPLICATE" not in reason_codes:
                reason_codes.append("DISCORD_SCOPE_RULE_DUPLICATE")
            continue
        seen_rules.add(fingerprint)
        normalized_rules.append(normalized_rule)
        if action == "allow":
            deny_only = False

    if errors:
        return None, reason_codes, errors

    if mode == "custom" and not normalized_rules:
        reason_codes.append("DISCORD_SCOPE_CUSTOM_EMPTY")
    if mode == "custom" and normalized_rules and deny_only:
        reason_codes.append("DISCORD_SCOPE_DENY_ONLY")

    return {
        "mode": mode,
        "effective_source": "ui",
        "rules": normalized_rules,
    }, reason_codes, []


def validate_settings(settings: Any) -> tuple[Optional[Dict[str, Any]], list[str], list[str]]:
    if not isinstance(settings, dict):
        return None, ["DISCORD_SETTINGS_PAYLOAD_INVALID"], ["settings payload must be an object"]

    reason_codes: list[str] = []
    errors: list[str] = []
    normalized: Dict[str, Any] = {}

    feature_flags = settings.get("feature_flags")
    if feature_flags is not None:
        if not isinstance(feature_flags, dict):
            return None, ["DISCORD_SETTINGS_PAYLOAD_INVALID"], ["feature_flags must be an object"]
        normalized_flags: Dict[str, Any] = {}
        for key in (
            "auto_clear_enabled",
            "context_summary_enabled",
            "no_response_reaction_enabled",
            "status_narration_enabled",
            "respond_with_reaction",
            "allowlist_admin_users_only",
        ):
            if key in feature_flags:
                if not isinstance(feature_flags[key], bool):
                    errors.append(f"{key} must be boolean")
                    continue
                normalized_flags[key] = bool(feature_flags[key])

        for key, minimum in (
            ("auto_clear_min_gap_sec", 60),
            ("context_summary_interval_sec", 60),
        ):
            if key in feature_flags:
                if not isinstance(feature_flags[key], int) or isinstance(feature_flags[key], bool) or int(feature_flags[key]) < minimum:
                    errors.append(f"{key} must be an integer >= {minimum}")
                    continue
                normalized_flags[key] = int(feature_flags[key])

        if "no_response_reaction_palette" in feature_flags:
            palette = feature_flags["no_response_reaction_palette"]
            if not isinstance(palette, list) or not all(str(item).strip() for item in palette):
                errors.append("no_response_reaction_palette must be a non-empty string array")
            else:
                normalized_flags["no_response_reaction_palette"] = [str(item).strip() for item in palette]

        if "status_narration_mode" in feature_flags:
            mode = str(feature_flags["status_narration_mode"]).strip().lower()
            if mode not in _STATUS_MODES:
                errors.append("status_narration_mode must be edit or send")
            else:
                normalized_flags["status_narration_mode"] = mode

        if "attachments" in feature_flags:
            attachments = feature_flags["attachments"]
            if not isinstance(attachments, dict):
                errors.append("attachments must be an object")
            else:
                normalized_attachments: Dict[str, Any] = {}
                for key in ("include_urls", "text_preview_enabled"):
                    if key in attachments:
                        if not isinstance(attachments[key], bool):
                            errors.append(f"attachments.{key} must be boolean")
                            continue
                        normalized_attachments[key] = bool(attachments[key])
                for key, minimum in (
                    ("text_preview_max_bytes", 1),
                    ("text_preview_max_chars", 80),
                ):
                    if key in attachments:
                        if not isinstance(attachments[key], int) or isinstance(attachments[key], bool) or int(attachments[key]) < minimum:
                            errors.append(f"attachments.{key} must be an integer >= {minimum}")
                            continue
                        normalized_attachments[key] = int(attachments[key])
                if normalized_attachments:
                    normalized_flags["attachments"] = normalized_attachments

        if normalized_flags:
            normalized["feature_flags"] = normalized_flags

    runtime = settings.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            return None, ["DISCORD_SETTINGS_PAYLOAD_INVALID"], ["runtime must be an object"]
        normalized_runtime: Dict[str, Any] = {}
        if "command_prefix" in runtime:
            raw_prefix = str(runtime["command_prefix"])
            if not raw_prefix.strip():
                errors.append("runtime.command_prefix must be a non-empty string")
            else:
                normalized_runtime["command_prefix"] = raw_prefix
        if "max_invoke_interval_sec" in runtime:
            value = runtime["max_invoke_interval_sec"]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.1:
                errors.append("runtime.max_invoke_interval_sec must be a number >= 0.1")
            else:
                normalized_runtime["max_invoke_interval_sec"] = float(value)
        if "anti_spam" in runtime:
            anti_spam = runtime["anti_spam"]
            if not isinstance(anti_spam, dict):
                errors.append("runtime.anti_spam must be an object")
            else:
                normalized_anti_spam: Dict[str, Any] = {}
                for key, minimum in (
                    ("window_sec", 1),
                    ("max_events", 0),
                    ("cooldown_sec", 0),
                ):
                    if key in anti_spam:
                        value = anti_spam[key]
                        if not isinstance(value, int) or isinstance(value, bool) or int(value) < minimum:
                            errors.append(f"runtime.anti_spam.{key} must be an integer >= {minimum}")
                            continue
                        normalized_anti_spam[key] = int(value)
                if normalized_anti_spam:
                    normalized_runtime["anti_spam"] = normalized_anti_spam
        if normalized_runtime:
            normalized["runtime"] = normalized_runtime

    if errors:
        return None, ["DISCORD_SETTINGS_VALUE_INVALID"], errors
    return normalized, reason_codes, []


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_saved_policy(context: Optional[Dict[str, Any]] = None) -> tuple[Optional[Dict[str, Any]], list[str]]:
    path = policy_path(context)
    if not path.exists():
        return None, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, ["DISCORD_POLICY_FILE_INVALID"]
    if not isinstance(raw, dict):
        return None, ["DISCORD_POLICY_FILE_INVALID"]
    return raw, []


def resolve_effective_policy(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    effective = default_policy()
    reason_codes: list[str] = []
    saved_policy, load_codes = load_saved_policy(context)
    reason_codes.extend(load_codes)
    if not isinstance(saved_policy, dict):
        return {
            "policy": effective,
            "reason_codes": reason_codes,
            "policy_path": str(policy_path(context)),
            "audit_path": str(audit_path(context)),
            "saved_policy_present": False,
        }

    saved_scope = saved_policy.get("scope")
    if saved_scope is not None:
        normalized_scope, scope_codes, scope_errors = validate_scope(saved_scope)
        reason_codes.extend(scope_codes)
        if not scope_errors and normalized_scope is not None:
            effective["scope"] = normalized_scope
            effective["scope"]["effective_source"] = "saved"
        else:
            if "DISCORD_POLICY_FILE_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_POLICY_FILE_INVALID")

    saved_settings = {}
    if isinstance(saved_policy.get("feature_flags"), dict):
        saved_settings["feature_flags"] = saved_policy["feature_flags"]
    if isinstance(saved_policy.get("runtime"), dict):
        saved_settings["runtime"] = saved_policy["runtime"]
    if saved_settings:
        normalized_settings, settings_codes, settings_errors = validate_settings(saved_settings)
        reason_codes.extend(settings_codes)
        if not settings_errors and normalized_settings is not None:
            effective = _deep_merge(effective, normalized_settings)
        else:
            if "DISCORD_POLICY_FILE_INVALID" not in reason_codes:
                reason_codes.append("DISCORD_POLICY_FILE_INVALID")

    if "enabled" in saved_policy:
        effective["enabled"] = bool(saved_policy["enabled"])

    unique_codes: list[str] = []
    for code in reason_codes:
        if code not in unique_codes:
            unique_codes.append(code)

    return {
        "policy": effective,
        "reason_codes": unique_codes,
        "policy_path": str(policy_path(context)),
        "audit_path": str(audit_path(context)),
        "saved_policy_present": True,
    }


def persist_policy(policy: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
    path = policy_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return True, None
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False, str(exc)


def append_ui_audit_event(
    *,
    action_id: str,
    method: str,
    context: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
    success: bool,
    mutated: bool,
    reason_codes: list[str],
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    record = {
        "recorded_at": _now_utc_iso(),
        "plugin_id": str((context or {}).get("plugin_id") or "busy-38-discord"),
        "actor": str((context or {}).get("actor") or (context or {}).get("user_id") or "unknown"),
        "action_id": str(action_id),
        "method": str(method).strip().upper() or "GET",
        "success": bool(success),
        "mutated": bool(mutated),
        "reason_codes": [str(code) for code in reason_codes if str(code).strip()],
        "payload_hash": _record_hash(payload or {}),
        "policy_path": str(policy_path(context)),
    }
    if before is not None:
        record["before_hash"] = _record_hash(before)
        record["before"] = before
    if after is not None:
        record["after_hash"] = _record_hash(after)
        record["after"] = after
    if error:
        record["error"] = str(error)

    target = audit_path(context)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        return True, None
    except Exception as exc:
        return False, str(exc)


def evaluate_scope(
    scope: Dict[str, Any],
    *,
    feature: str,
    guild_id: Optional[int] = None,
    channel_id: Optional[int] = None,
    role_ids: Optional[list[int]] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    mode = str(scope.get("mode") or "").strip().lower()
    if mode == "all":
        return {"allowed": True, "reason_code": "DISCORD_SCOPE_ALLOW_ALL", "matched_rule": None}
    if mode == "none":
        return {"allowed": False, "reason_code": "DISCORD_SCOPE_DISABLED", "matched_rule": None}

    normalized_feature = str(feature or "").strip().lower()
    candidates: list[tuple[int, int, Dict[str, Any]]] = []
    for index, raw_rule in enumerate(scope.get("rules") or []):
        if not isinstance(raw_rule, dict):
            continue
        features = raw_rule.get("features") or []
        if features and normalized_feature not in features:
            continue
        target = str(raw_rule.get("target") or "").strip().lower()
        ids = [str(item) for item in (raw_rule.get("ids") or [])]
        matched = False
        if target == "guild" and guild_id is not None and str(guild_id) in ids:
            matched = True
        elif target == "channel" and channel_id is not None and str(channel_id) in ids:
            matched = True
        elif target == "user" and user_id is not None and str(user_id) in ids:
            matched = True
        elif target == "role" and role_ids:
            matched = any(str(role_id) in ids for role_id in role_ids)
        if not matched:
            continue
        specificity = _SCOPE_SPECIFICITY.get(target, 0)
        deny_rank = 0 if str(raw_rule.get("action") or "").strip().lower() == "deny" else 1
        candidates.append((specificity, deny_rank, raw_rule))

    if not candidates:
        return {"allowed": False, "reason_code": "DISCORD_SCOPE_NO_MATCH", "matched_rule": None}

    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = candidates[0][2]
    allowed = str(chosen.get("action") or "").strip().lower() == "allow"
    return {
        "allowed": allowed,
        "reason_code": "DISCORD_SCOPE_RULE_ALLOW" if allowed else "DISCORD_SCOPE_RULE_DENY",
        "matched_rule": deepcopy(chosen),
    }
