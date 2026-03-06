# Discord Debug Method Enforcement Change Request

Status: Approved for implementation (2026-03-05)
Owner: platform/ui

## Problem

The repo-local UI contract declares the Discord `debug` action as `GET`, but the implementation
currently accepts other verbs as successful requests.

That creates contract drift in an authority path:

- `ui/manifest.json` advertises a literal verb boundary,
- the handler accepts out-of-contract methods,
- callers can observe successful behavior that the declared UI contract never authorized.

## Decision

The Discord plugin-local `debug` action is `GET`-only.

Any non-`GET` invocation must:

- fail closed,
- return `success: false`,
- emit `DISCORD_UI_METHOD_INVALID`,
- append a local audit record describing the rejected invocation.

## Scope

In scope:

- `ui/actions.py` method enforcement for `handle_debug`,
- unit coverage for rejected non-`GET` calls,
- spec/current-state updates describing the literal verb contract.

Out of scope:

- changing the shape of scope/settings/validate actions,
- changing persisted policy semantics.

## Acceptance criteria

- `GET /ui/debug` remains successful,
- non-`GET` `debug` invocations fail closed with `DISCORD_UI_METHOD_INVALID`,
- rejected calls are still audited locally,
- tests cover both accepted and rejected paths.
