# busy38-discord API Reference

This plugin exposes agent-facing capabilities via Busy38 cheatcodes.

## Namespace: `dlog`

Transcript tools backed by Busy38’s DuckDB `chat_entries` table (created by the Discord integration).

### `dlog:search`

Broad search with snippet windows around matches. Defaults to recency (last 24h).

Example:
```text
[dlog:search query="deploy failed" project_id="discord:123:456" max_age_hours=24 /]
```

Parameters:
- `query` (string, required): literal unless `regex=true`
- `project_id` (string, optional): `discord:<guild_id>:<channel_id>`
- `max_age_hours` (int, default 24): 0 disables age filter
- `max_messages` (int, default 5000): max rows scanned (newest-first)
- `context` (int, default 80): context window in characters for each snippet
- `case_sensitive` (bool, default false)
- `regex` (bool, default false)
- `snippets_per_message` (int, default 2)
- `max_results` (int, default 20)

Returns:
- `{success: true, results: [...]}` where each result includes:
  - `id`: `discord:<message_id>`
  - `timestamp`
  - `project_id`
  - `snippets`: list of strings
  - `metadata`: author/channel ids, etc.

### `dlog:around`

Fetches surrounding messages for one transcript entry.

Example:
```text
[dlog:around message_id="123456789012345678" before=8 after=8 /]
```

Parameters:
- `message_id` (string, required): numeric ID or `discord:<id>`
- `before` (int, default 8)
- `after` (int, default 8)

Returns:
- `{success: true, rows: [...]}` where each row has `content` + `metadata`.

## Namespace: `dforum`

Forum/thread operations. These are async and require that Busy38’s Discord bot has set the active runtime bot instance.

### `dforum:reply`

Post a message in a thread:
```text
[dforum:reply thread_id=123 content="Status: investigating" /]
```

### `dforum:set_tags`

Apply forum tags to a thread by name:
```text
[dforum:set_tags thread_id=123 tag_names="In Progress,Blocked" /]
```

### `dforum:rename`

Rename a thread:
```text
[dforum:rename thread_id=123 name="Task: Fix deploy pipeline" /]
```

### `dforum:archive`

Archive/unarchive a thread:
```text
[dforum:archive thread_id=123 archived=true /]
```

### `dforum:lock`

Lock/unlock a thread:
```text
[dforum:lock thread_id=123 locked=true /]
```

### `dforum:create_post`

Create a new forum post (thread):
```text
[dforum:create_post forum_id=999 title="Task: Audit errors" content="Please investigate..." tag_names="New" /]
```

### `dforum:get_thread`

Fetch recent messages and applied tags:
```text
[dforum:get_thread thread_id=123 limit=30 /]
```

## Security Notes

- `dlog:*` reads local chat logs only; it does not fetch from Discord.
- `dforum:*` uses Discord APIs and should be treated as a privileged capability.

