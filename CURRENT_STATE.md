# busy38-discord Current State

Last Updated: 2026-03-05

## Current behavior

- Discord transcript search and relay tooling remains the core shipped behavior of this plugin.
- Discord runtime behavior is still primarily seeded from environment variables and SquidKeys-backed
  authorization inputs.
- Plugin-local management UI support now includes a documented admin policy surface under `ui/`
  for scope, settings, validation, and debug actions.

## New constraints

- Plugin-local UI actions must remain fail-closed on malformed scope/settings payloads.
- UI actions must leave a visible local audit trail for every invocation.
- Persisted Discord policy is non-secret configuration only and must not be used to store tokens or
  other credentials.
- Admin-configured command prefixes are persisted verbatim; whitespace-only prefixes are rejected.

## Expected files

- `ui/manifest.json`
- `ui/actions.py`
- `toolkit/discord_policy.py`
- `tests/test_discord_ui_actions.py`
