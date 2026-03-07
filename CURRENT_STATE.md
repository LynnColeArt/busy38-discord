# busy38-discord Current State

Last Updated: 2026-03-07

## Current behavior

- Discord transcript search and relay tooling remains the core shipped behavior of this plugin.
- Discord runtime behavior is still primarily seeded from environment variables and SquidKeys-backed
  authorization inputs.
- Plugin-local management UI support now includes a documented admin policy surface under `ui/`
  for scope, settings, validation, and debug actions.

## New constraints

- Plugin-local UI actions must remain fail-closed on malformed scope/settings payloads.
- The plugin-local `debug` action is `GET`-only and must reject any other method with
  `DISCORD_UI_METHOD_INVALID`.
- UI actions must leave a visible local audit trail for every invocation.
- Mutating UI actions must reject malformed non-boolean `enabled` values and
  must not report success if the audit sink cannot record the mutation.
- Read/preview admin actions must also fail visibly if the audit sink cannot
  record the invocation:
  - settings reads do not report success without an audit record,
  - validation preview does not report success without an audit record.
- Saved policy load must also validate persisted `enabled` literally:
  - malformed saved values emit explicit invalid-policy reason codes,
  - the affected enabled path fails closed instead of being re-enabled by truthiness.
- The settings/validate UI contract now round-trips `enabled` literally:
  - `GET /ui/settings` returns the effective persisted `enabled` value,
  - validation preview reflects candidate `enabled` changes instead of hiding
    them behind the default policy value.
- Persisted Discord policy is non-secret configuration only and must not be used to store tokens or
  other credentials.
- Admin-configured command prefixes are persisted verbatim; whitespace-only prefixes are rejected.

## Expected files

- `docs/internal/DISCORD_DEBUG_METHOD_ENFORCEMENT_CHANGE_REQUEST.md`
- `ui/manifest.json`
- `ui/actions.py`
- `toolkit/discord_policy.py`
- `tests/test_discord_ui_actions.py`
