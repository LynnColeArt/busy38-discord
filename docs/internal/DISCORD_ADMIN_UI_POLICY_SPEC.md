# Discord Admin UI Policy Spec

Status: Approved for implementation (2026-03-05)
Owner: platform/ui

## Purpose

This repo must expose the plugin-local Discord admin policy surface already referenced by the main
Busy roadmap and change-request records.

The scope of this repo-local implementation is:

- plugin-owned UI manifest and action handlers,
- deterministic policy validation,
- plugin-local policy persistence,
- plugin-local UI action audit logging,
- debug visibility for the effective Discord policy state.

## Authority boundary

- The management plane remains the only valid caller for plugin-local admin actions.
- Plugin-local handlers validate payloads and persist non-secret policy only.
- Invalid scope or settings payloads must fail closed with explicit reason codes.
- UI action invocations must emit an append-only local audit record.
- Read/preview handlers must fail visibly if the required audit append cannot be
  written; they must not report unqualified success without that audit record.

## Policy model

Persist a plugin-local `discord_policy` document with three top-level sections:

1. `enabled`
2. `scope`
3. `feature_flags`
4. `runtime`

### Scope

`scope.mode` must be one of:

- `all`
- `custom`
- `none`

`scope.rules` entries must be explicit objects:

- `target`: `guild|channel|role|user`
- `ids`: non-empty list of numeric Discord IDs
- `action`: `allow|deny`
- `features`: optional subset of `commands|admin|auto_clear|context_summary|attachments|status`

Rules:

- `custom` with no rules is deny-all by design.
- `deny` takes precedence over `allow` at equal specificity.
- More specific targets outrank less specific ones:
  `user > role > channel > guild`.

## UI action contract

Handlers live in `ui/actions.py`.

Supported actions:

- `GET /ui/debug`
- `GET /ui/scope`
- `POST /ui/scope`
- `GET /ui/settings`
- `POST /ui/settings`
- `POST /ui/validate`

Action handlers use the standard signature:

`handle_<action>(payload: dict | None, method: str, context: dict | None) -> dict`

Verb enforcement is literal:

- `debug` is `GET`-only and must reject any other method with
  `DISCORD_UI_METHOD_INVALID`,
- rejected calls must still produce a local audit record.

## Persistence

- Policy path:
  - context override `policy_path`, else
  - env `BUSY38_DISCORD_POLICY_PATH`, else
  - `<plugin_root>/data/discord_policy.json`
- Audit path:
  - context override `audit_path`, else
  - env `BUSY38_DISCORD_UI_AUDIT_PATH`, else
  - `<plugin_root>/data/discord_ui_actions.ndjson`

Persisted policy must be JSON and contain no credentials.

## Settings normalization

- `runtime.command_prefix` must be rejected if it is empty or whitespace-only.
- Otherwise `runtime.command_prefix` must be persisted exactly as provided so the
  configured invocation formatting remains explicit and auditable.

## Required reason codes

At minimum:

- `DISCORD_UI_METHOD_INVALID`
- `DISCORD_SCOPE_PAYLOAD_INVALID`
- `DISCORD_SCOPE_MODE_INVALID`
- `DISCORD_SCOPE_RULE_TARGET_INVALID`
- `DISCORD_SCOPE_RULE_ACTION_INVALID`
- `DISCORD_SCOPE_RULE_IDS_INVALID`
- `DISCORD_SCOPE_RULE_FEATURE_INVALID`
- `DISCORD_SCOPE_RULE_DUPLICATE`
- `DISCORD_SCOPE_CUSTOM_EMPTY`
- `DISCORD_SETTINGS_PAYLOAD_INVALID`
- `DISCORD_SETTINGS_VALUE_INVALID`
- `DISCORD_POLICY_FILE_INVALID`
- `DISCORD_POLICY_PERSIST_FAILED`
- `DISCORD_POLICY_AUDIT_FAILED`

## Validation requirements

- Scope resolution precedence must be unit-tested.
- Invalid scope/settings payloads must return deterministic reason codes.
- Successful writes must be visible in subsequent debug reads.
- UI audit output must be produced for action invocations.
- Read/preview actions must not report success if their required audit append
  fails.
