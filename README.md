# busy38-discord

Busy38 vendor plugin that provides the full Discord transport surface:
- Discord runtime bot (`toolkit/discord_bot.py`)
- Discord state/transcript/runtime bridge modules
- Agent cheatcodes for transcript search and forum task-board operations

This plugin is intended to be **vendored into Busy38** under `./vendor/busy-38-discord/` and loaded by Busy38’s `PluginManager`.

## What It Adds

### Runtime

- `Busy38DiscordBot` now lives in this plugin (`toolkit/discord_bot.py`).
- Busy core keeps thin compatibility shims under `integrations/discord_*.py`.
- The bot supports:
  - full-channel ingestion + follow mode
  - forum task-board subscriptions
  - recency-biased context (24h default)
  - no-response reaction acknowledgements (emoji palette + optional agent directive)
  - anti-spam follow guardrails for high-traffic channels
  - assignment/handoff-aware multi-agent coordination hints
  - hardware-auth enforcement hooks

### Agent Cheatcodes

The plugin registers these cheatcode namespaces in Busy38’s cheatcode registry:

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
- `DISCORD_CONTEXT_MAX_AGE_SEC`: recency bias for prompt context (default `86400`)
- `DISCORD_NO_RESPONSE_REACTIONS`: enable reaction acknowledgement for silent decisions (default `1`)
- `DISCORD_NO_RESPONSE_REACTIONS_ON_BOTS`: allow silent reactions on bot-authored messages (default `1`)
- `DISCORD_NO_RESPONSE_EMOJIS`: comma-separated default emoji palette (default `👍,👀,✅`)
- `DISCORD_FOLLOW_SPAM_WINDOW_SEC`: follow-mode guardrail window seconds (default `30`)
- `DISCORD_FOLLOW_SPAM_MAX_EVENTS`: max follow triggers in window before cooldown (default `12`)
- `DISCORD_FOLLOW_SPAM_COOLDOWN_SEC`: cooldown after burst limit is exceeded (default `45`)

## Licensing

GPL-3.0-only. See `LICENSE`.
