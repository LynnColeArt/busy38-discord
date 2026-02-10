# busy38-discord

Busy38 vendor plugin that turns Discord forums into a usable task board surface for agents, and exposes agent-internal transcript search tools.

This plugin is intended to be **vendored into Busy38** under `./vendor/busy-38-discord/` and loaded by Busy38’s `PluginManager`.

## What It Adds

### Agent Cheatcodes

The plugin registers these cheatcode namespaces into Busy38’s cheatcode registry:

- `dlog:*`: search Busy38’s Discord transcript DB (broad search with snippets, then drill-down context)
- `dforum:*`: async operations for forum threads (reply, rename, tag, archive/lock, create posts)

See `API_REFERENCE.md` and `tool_spec.yaml`.

## Requirements

- Busy38 host runtime (this plugin imports Busy38’s cheatcode registry).
- `discord.py` in the Busy38 environment.
- DuckDB chat logs created by Busy38’s Discord integration:
  - default location: `./data/memory/chat_logs.duckdb`

## Installation (Vendored)

1. Place this repo at `Busy38/vendor/busy-38-discord/`.
2. Ensure Busy38 loads vendor plugins (Busy38 `PluginManager` scans `./vendor`).
3. Start the Busy38 Discord bot; the bot sets a runtime pointer used by `dforum:*` actions.

## Configuration

- `BUSY38_CHATLOG_DIR`: directory containing `chat_logs.duckdb` (default `./data/memory`)

## Licensing

GPL-3.0-only. See `LICENSE`.

