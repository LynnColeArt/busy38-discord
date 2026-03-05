from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# The host may import this module directly from file path, so make the plugin
# root importable before resolving shared toolkit helpers.
_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

_POLICY_SPEC = importlib.util.spec_from_file_location(
    "busy38_discord_policy",
    _PLUGIN_ROOT / "toolkit" / "discord_policy.py",
)
assert _POLICY_SPEC is not None and _POLICY_SPEC.loader is not None
_POLICY_MODULE = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(_POLICY_MODULE)

append_ui_audit_event = _POLICY_MODULE.append_ui_audit_event
audit_path = _POLICY_MODULE.audit_path
evaluate_scope = _POLICY_MODULE.evaluate_scope
persist_policy = _POLICY_MODULE.persist_policy
policy_path = _POLICY_MODULE.policy_path
resolve_effective_policy = _POLICY_MODULE.resolve_effective_policy
validate_scope = _POLICY_MODULE.validate_scope
validate_settings = _POLICY_MODULE.validate_settings


def _normalized_context(context: dict | None) -> dict:
    return dict(context or {})


def _extract_scope_payload(payload: dict | None) -> Any:
    payload_dict = dict(payload or {})
    if set(payload_dict.keys()) <= {"mode", "rules"}:
        return payload_dict
    if set(payload_dict.keys()) == {"scope"}:
        return payload_dict.get("scope")
    return None


def _extract_settings_payload(payload: dict | None) -> Any:
    payload_dict = dict(payload or {})
    if set(payload_dict.keys()) <= {"feature_flags", "runtime", "enabled"}:
        return payload_dict
    if set(payload_dict.keys()) == {"settings"}:
        return payload_dict.get("settings")
    return None


def _build_debug_payload(method: str, context: dict | None) -> dict:
    normalized_context = _normalized_context(context)
    source_path = str(normalized_context.get("source_path") or "").strip()
    plugin_id = str(normalized_context.get("plugin_id") or "busy-38-discord").strip()
    plugin_root = Path(source_path).resolve() if source_path else _PLUGIN_ROOT
    resolved = resolve_effective_policy(normalized_context)
    return {
        "plugin": plugin_id,
        "method": str(method).strip().upper() or "GET",
        "source_path": str(plugin_root),
        "source_exists": plugin_root.exists(),
        "manifests": {
            "plugin_manifest_exists": (plugin_root / "manifest.json").is_file(),
            "ui_manifest_exists": (plugin_root / "ui" / "manifest.json").is_file(),
        },
        "paths": {
            "policy_path": str(policy_path(normalized_context)),
            "audit_path": str(audit_path(normalized_context)),
        },
        "policy": resolved["policy"],
        "reason_codes": resolved["reason_codes"],
        "scope_preview": {
            "commands": evaluate_scope(
                resolved["policy"]["scope"],
                feature="commands",
                guild_id=1,
                channel_id=2,
                role_ids=[3],
                user_id=4,
            ),
        },
        "entrypoint": "actions:handle_debug",
    }


def _audit(
    *,
    action_id: str,
    method: str,
    context: dict | None,
    payload: dict | None,
    success: bool,
    mutated: bool,
    reason_codes: list[str],
    before: Optional[dict] = None,
    after: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    append_ui_audit_event(
        action_id=action_id,
        method=method,
        context=context,
        payload=payload,
        success=success,
        mutated=mutated,
        reason_codes=reason_codes,
        before=before,
        after=after,
        error=error,
    )


def handle_debug(payload: dict | None, method: str, context: dict | None) -> dict:
    debug_payload = _build_debug_payload(method, context)
    _audit(
        action_id="debug",
        method=method,
        context=context,
        payload=payload,
        success=True,
        mutated=False,
        reason_codes=list(debug_payload.get("reason_codes") or []),
    )
    return {
        "success": True,
        "message": "plugin ui debug handler executed",
        "payload": debug_payload,
    }


def handle_scope(payload: dict | None, method: str, context: dict | None) -> dict:
    normalized_method = str(method).strip().upper() or "GET"
    resolved = resolve_effective_policy(context)
    current_policy = resolved["policy"]
    current_scope = current_policy["scope"]
    if normalized_method == "GET":
        _audit(
            action_id="scope",
            method=normalized_method,
            context=context,
            payload=payload,
            success=True,
            mutated=False,
            reason_codes=list(resolved["reason_codes"] or []),
        )
        return {
            "success": True,
            "message": "discord scope policy loaded",
            "payload": {
                "scope": current_scope,
                "reason_codes": resolved["reason_codes"],
            },
        }

    if normalized_method != "POST":
        reason_codes = ["DISCORD_SCOPE_METHOD_INVALID"]
        _audit(
            action_id="scope",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error="scope action only supports GET and POST",
        )
        return {
            "success": False,
            "message": "invalid scope method",
            "reason_codes": reason_codes,
        }

    candidate_scope = _extract_scope_payload(payload)
    normalized_scope, reason_codes, errors = validate_scope(candidate_scope)
    if errors or normalized_scope is None:
        _audit(
            action_id="scope",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error="; ".join(errors),
        )
        return {
            "success": False,
            "message": "invalid discord scope payload",
            "reason_codes": reason_codes,
            "errors": errors,
        }

    updated_policy = dict(current_policy)
    updated_policy["scope"] = normalized_scope
    ok, persist_error = persist_policy(updated_policy, context)
    if not ok:
        reason_codes = list(reason_codes) + ["DISCORD_POLICY_PERSIST_FAILED"]
        _audit(
            action_id="scope",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error=persist_error,
        )
        return {
            "success": False,
            "message": "failed to persist discord scope policy",
            "reason_codes": reason_codes,
            "errors": [str(persist_error or "unknown persistence failure")],
        }

    _audit(
        action_id="scope",
        method=normalized_method,
        context=context,
        payload=payload,
        success=True,
        mutated=True,
        reason_codes=reason_codes,
        before=current_policy,
        after=updated_policy,
    )
    return {
        "success": True,
        "message": "discord scope policy updated",
        "payload": {
            "scope": updated_policy["scope"],
            "reason_codes": reason_codes,
        },
    }


def handle_settings(payload: dict | None, method: str, context: dict | None) -> dict:
    normalized_method = str(method).strip().upper() or "GET"
    resolved = resolve_effective_policy(context)
    current_policy = resolved["policy"]
    if normalized_method == "GET":
        _audit(
            action_id="settings",
            method=normalized_method,
            context=context,
            payload=payload,
            success=True,
            mutated=False,
            reason_codes=list(resolved["reason_codes"] or []),
        )
        return {
            "success": True,
            "message": "discord settings loaded",
            "payload": {
                "settings": {
                    "feature_flags": current_policy["feature_flags"],
                    "runtime": current_policy["runtime"],
                },
                "reason_codes": resolved["reason_codes"],
            },
        }

    if normalized_method != "POST":
        reason_codes = ["DISCORD_SETTINGS_METHOD_INVALID"]
        _audit(
            action_id="settings",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error="settings action only supports GET and POST",
        )
        return {
            "success": False,
            "message": "invalid settings method",
            "reason_codes": reason_codes,
        }

    candidate_settings = _extract_settings_payload(payload)
    normalized_settings, reason_codes, errors = validate_settings(candidate_settings)
    if errors or normalized_settings is None:
        _audit(
            action_id="settings",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error="; ".join(errors),
        )
        return {
            "success": False,
            "message": "invalid discord settings payload",
            "reason_codes": reason_codes,
            "errors": errors,
        }

    updated_policy = dict(current_policy)
    if "enabled" in (candidate_settings or {}):
        updated_policy["enabled"] = bool((candidate_settings or {})["enabled"])
    if "feature_flags" in normalized_settings:
        updated_policy["feature_flags"] = {
            **current_policy["feature_flags"],
            **normalized_settings["feature_flags"],
        }
        if "attachments" in normalized_settings["feature_flags"]:
            updated_policy["feature_flags"]["attachments"] = {
                **current_policy["feature_flags"]["attachments"],
                **normalized_settings["feature_flags"]["attachments"],
            }
    if "runtime" in normalized_settings:
        updated_policy["runtime"] = {
            **current_policy["runtime"],
            **normalized_settings["runtime"],
        }
        if "anti_spam" in normalized_settings["runtime"]:
            updated_policy["runtime"]["anti_spam"] = {
                **current_policy["runtime"]["anti_spam"],
                **normalized_settings["runtime"]["anti_spam"],
            }

    ok, persist_error = persist_policy(updated_policy, context)
    if not ok:
        reason_codes = list(reason_codes) + ["DISCORD_POLICY_PERSIST_FAILED"]
        _audit(
            action_id="settings",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error=persist_error,
        )
        return {
            "success": False,
            "message": "failed to persist discord settings",
            "reason_codes": reason_codes,
            "errors": [str(persist_error or "unknown persistence failure")],
        }

    _audit(
        action_id="settings",
        method=normalized_method,
        context=context,
        payload=payload,
        success=True,
        mutated=True,
        reason_codes=reason_codes,
        before=current_policy,
        after=updated_policy,
    )
    return {
        "success": True,
        "message": "discord settings updated",
        "payload": {
            "settings": {
                "feature_flags": updated_policy["feature_flags"],
                "runtime": updated_policy["runtime"],
            },
            "reason_codes": reason_codes,
        },
    }


def handle_validate(payload: dict | None, method: str, context: dict | None) -> dict:
    normalized_method = str(method).strip().upper() or "POST"
    resolved = resolve_effective_policy(context)
    current_policy = resolved["policy"]
    if normalized_method != "POST":
        reason_codes = ["DISCORD_VALIDATE_METHOD_INVALID"]
        _audit(
            action_id="validate",
            method=normalized_method,
            context=context,
            payload=payload,
            success=False,
            mutated=False,
            reason_codes=reason_codes,
            before=current_policy,
            error="validate action only supports POST",
        )
        return {
            "success": False,
            "message": "invalid validate method",
            "reason_codes": reason_codes,
        }

    payload_dict = dict(payload or {})
    reason_codes: list[str] = []
    errors: list[str] = []
    preview_policy = dict(current_policy)

    if "scope" in payload_dict:
        normalized_scope, scope_codes, scope_errors = validate_scope(payload_dict.get("scope"))
        reason_codes.extend(scope_codes)
        errors.extend(scope_errors)
        if normalized_scope is not None and not scope_errors:
            preview_policy["scope"] = normalized_scope

    if "settings" in payload_dict:
        normalized_settings, settings_codes, settings_errors = validate_settings(payload_dict.get("settings"))
        reason_codes.extend(settings_codes)
        errors.extend(settings_errors)
        if normalized_settings is not None and not settings_errors:
            if "feature_flags" in normalized_settings:
                preview_policy["feature_flags"] = {
                    **preview_policy["feature_flags"],
                    **normalized_settings["feature_flags"],
                }
                if "attachments" in normalized_settings["feature_flags"]:
                    preview_policy["feature_flags"]["attachments"] = {
                        **preview_policy["feature_flags"]["attachments"],
                        **normalized_settings["feature_flags"]["attachments"],
                    }
            if "runtime" in normalized_settings:
                preview_policy["runtime"] = {
                    **preview_policy["runtime"],
                    **normalized_settings["runtime"],
                }
                if "anti_spam" in normalized_settings["runtime"]:
                    preview_policy["runtime"]["anti_spam"] = {
                        **preview_policy["runtime"]["anti_spam"],
                        **normalized_settings["runtime"]["anti_spam"],
                    }

    success = not errors
    _audit(
        action_id="validate",
        method=normalized_method,
        context=context,
        payload=payload,
        success=success,
        mutated=False,
        reason_codes=reason_codes,
        before=current_policy,
        after=preview_policy if success else None,
        error="; ".join(errors) if errors else None,
    )
    return {
        "success": success,
        "message": "discord policy validation complete" if success else "discord policy validation failed",
        "reason_codes": reason_codes,
        "errors": errors,
        "payload": {
            "policy_preview": preview_policy,
        },
    }
