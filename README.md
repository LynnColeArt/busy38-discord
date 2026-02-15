# busy38-discord

Busy38 vendor plugin that provides the full Discord transport surface:
- Discord runtime bot (`toolkit/discord_bot.py`)
- Discord state/transcript/runtime bridge modules
- Agent cheatcodes for transcript search and forum task-board operations

Roadmap note: full internal rebrand Phase 2 is deferred until after closed-beta hardening is complete.

This plugin is intended to be **vendored into Busy38** under `./vendor/busy-38-discord/` and loaded by Busy38’s `PluginManager`.

## What It Adds

### Runtime

- `Busy38DiscordBot` now lives in this plugin (`toolkit/discord_bot.py`).
- Busy core imports this runtime directly from `vendor/busy-38-discord/toolkit`.
- The bot supports:
  - full-channel ingestion + follow mode
  - attachment-aware ingestion (metadata + optional text previews)
  - forum task-board subscriptions
  - recency-biased context (24h default)
  - attachment directives in agent replies (`[attach ... /]`, including base64/data URI payloads)
  - no-response reaction acknowledgements (emoji palette + optional agent directive)
  - anti-spam follow guardrails for high-traffic channels
  - assignment/handoff-aware multi-agent coordination hints
  - hardware-auth enforcement hooks
  - moderated `!busy38 clear [hours]` summary+pin context reset flow
  - optional heartbeat-driven auto-clear job registration via hooks
  - optional status narration messages during work (action-style)

### Agent Cheatcodes

The plugin registers these cheatcode namespaces in Busy38’s cheatcode registry:

- `dlog:*`: search Busy38’s Discord transcript DB (broad search with snippets, then drill-down context)
- `dforum:*`: async operations for forum threads (reply, rename, tag, archive/lock, create posts)
  - `reply` and `create_post` support attachments from local paths, URLs, base64 payloads, or data URIs
- `drelay:*`: agent relay bus for coordinated multi-agent workflows without direct Discord channel permissions
  - `emit` posts structured room-scoped updates
  - `read` retrieves recent room context and status updates
  - `status` returns per-room backlog counts

See `API_REFERENCE.md` and `tool_spec.yaml`.

## AI-Generated / Automated Contributions

Automated code generation and AI-assisted submissions are welcome.

For production code, placeholders are not acceptable.

- Unit tests may use mocks and stubs.
- Runtime code must be functional and complete when merged.
- New functionality must include unit tests (or updates to existing tests) that cover the behavior.

Before submitting generated changes, verify:

- No functional file includes temporary placeholders (`TODO`, `FIXME`, `NotImplementedError`).
- Mock/stub logic is confined to tests and fixtures.
- New behavior has tests or integration checks and explicit edge-case handling.
- Changes do not introduce silent fallbacks or remove validation/security paths.
- Behavior changes are documented where needed (plugin docs, release notes, or API references).
- All relevant tests pass before merge.

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
- `DISCORD_CLEAR_WINDOW_HOURS`: summary window for `!busy38 clear` and auto-clear (default `72`)
- `DISCORD_CLEAR_MAX_MESSAGES`: transcript message cap for summary generation (default `1200`)
- `DISCORD_AUTO_CLEAR_ENABLE`: register heartbeat auto-clear job when `1` (default `0`)
- `DISCORD_AUTO_CLEAR_INTERVAL_SEC`: heartbeat job interval for auto-clear checks (default `900`)
- `DISCORD_AUTO_CLEAR_MIN_GAP_SEC`: per-channel minimum time between clear operations (default `21600`)
- `DISCORD_CLEAR_STATE_PATH`: local file for per-channel clear timestamps (default `./data/discord_clear_state.json`)
- `DISCORD_AGENT_ATTACHMENTS_ENABLE`: parse/send `[attach ... /]` directives from agent output (default `1`)
- `DISCORD_ATTACHMENT_MAX_BYTES`: max file size per outbound attachment in bytes (default `8000000`)
- `DISCORD_ATTACHMENT_MAX_FILES`: max attachments per outbound message (default `10`)
- `DISCORD_ATTACHMENT_INCLUDE_URLS`: append attachment URLs into ingested content lines (default `1`)
- `DISCORD_ATTACHMENT_TEXT_PREVIEW_ENABLE`: attempt text previews for small text-like files (default `1`)
- `DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_BYTES`: max inbound file size for preview extraction (default `65536`)
- `DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_CHARS`: max preview characters saved per attachment (default `1200`)
- `DISCORD_RELAY_ROOM_MAX_EVENTS`: per-room relay backlog cap (default `200`)
- `DISCORD_RELAY_GLOBAL_MAX_EVENTS`: global relay backlog cap across all rooms (default `5000`)
- `DISCORD_STATUS_ENABLE`: enable action-style status messages during work (default `0`)
- `DISCORD_STATUS_MODE`: `edit` (single message updated) or `message` (new message per update) (default `edit`)
- `DISCORD_STATUS_STYLE`: `implicit` (like `/me`, do not repeat bot name in message body) or `explicit` (include name) (default `implicit`)
- `DISCORD_STATUS_DELAY_SEC`: delay before posting initial “thinking” status (default `1.5`)
- `DISCORD_STATUS_MIN_INTERVAL_SEC`: minimum time between status edits/sends per channel (default `2.5`)
- `DISCORD_STATUS_DELETE_ON_FINISH`: delete status message after response (default `1`)
- `DISCORD_STATUS_INCLUDE_ARGS`: include tool args in narration (reserved; default `0`)

## Licensing

GPL-3.0-only. See `LICENSE`.
