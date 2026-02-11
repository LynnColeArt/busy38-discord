"""
Discord Bot Integration for Busy38

Allows chatting with Busy38 agents directly in Discord.
Uses SquidKeys for secure credential storage.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import discord
    from discord.ext import commands
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None
    commands = None

from core.orchestration.integration import Busy38Orchestrator, OrchestratorConfig
from core.plugins.manager import PluginManager

# Discord "citizen" state
from .discord_state import (
    ChannelKey,
    MessageRecord,
    DiscordStateStore,
    load_subscriptions_from_keystore,
    save_subscriptions_to_keystore,
)
from .discord_transcript import DiscordTranscriptLogger

# Hardware-bound authorization gate
from core.security.hardware_auth import require_hardware_auth, hardware_auth_is_configured, HardwareAuthError

# Runtime bridge for vendor discord tools (dforum)
from .discord_runtime import set_bot as _set_discord_runtime_bot

# SquidKeys integration
try:
    import sys
    _candidates = [Path(__file__).resolve().parents[i] for i in range(1, 8)]
    for base in _candidates:
        squidkeys_path = base / "SquidKeys" / "src"
        if squidkeys_path.exists():
            sys.path.insert(0, str(squidkeys_path))
            break
    from key_store import KeyStore
    HAS_SQUIDKEYS = True
except ImportError:
    HAS_SQUIDKEYS = False
    KeyStore = None

logger = logging.getLogger(__name__)


class Busy38DiscordBot:
    """Discord bot that routes messages to Busy38 agents."""
    
    def __init__(self, 
                 token: Optional[str] = None, 
                 orchestrator: Optional[Busy38Orchestrator] = None,
                 allowed_users: Optional[list] = None,
                 use_squidkeys: bool = True):
        """Initialize Discord bot.
        
        Args:
            token: Discord bot token (or from DISCORD_TOKEN env var or SquidKeys)
            orchestrator: Busy38 orchestrator instance (creates one if not provided)
            allowed_users: List of allowed Discord user IDs for DMs (None = allow all)
            use_squidkeys: Whether to use SquidKeys for credential storage
        """
        if not HAS_DISCORD:
            raise ImportError("discord.py not installed. Run: pip install discord.py")
        
        # Initialize SquidKeys if available
        self.keystore = None
        if use_squidkeys and HAS_SQUIDKEYS:
            try:
                db_path = os.getenv("SQUIDKEYS_DB_PATH", "./data/keystore.duckdb")
                self.keystore = KeyStore(db_path=db_path)
                logger.info(f"SquidKeys initialized: {db_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize SquidKeys: {e}")
        
        # Get token from SquidKeys, env var, or parameter
        self.token = token or self._get_token_from_squidkeys() or os.getenv("DISCORD_TOKEN")
        if not self.token:
            raise ValueError(
                "Discord token required.\n"
                "Options:\n"
                "  1. Store in SquidKeys: save_authorization(agent_id='busy38', provider='discord', ...)\n"
                "  2. Set DISCORD_TOKEN environment variable\n"
                "  3. Pass token parameter"
            )
        
        # User allowlist for DMs (security) - from SquidKeys, env, or parameter
        self.allowed_users = self._get_allowed_users_from_squidkeys() or self._parse_allowed_users_env() or (set(allowed_users) if allowed_users else None)
        
        # Create or use provided orchestrator
        if orchestrator:
            self.orchestrator = orchestrator
            self._owns_orchestrator = False
        else:
            config = OrchestratorConfig(max_iterations=10)
            self.orchestrator = Busy38Orchestrator(config=config)
            self._owns_orchestrator = True
        
        # Discord channel state + subscription model
        self.state = DiscordStateStore(
            default_history_limit=int(os.getenv("DISCORD_HISTORY_LIMIT", "80"))
        )
        self._sub_file = Path(os.getenv("DISCORD_SUBSCRIPTIONS_PATH", "./data/discord_subscriptions.json"))
        self._load_subscriptions()

        # Persist all observed messages into DuckDB chat logs (for "boards"/analysis)
        self.transcript = DiscordTranscriptLogger(data_dir=os.getenv("BUSY38_CHATLOG_DIR", "./data/memory"))
        try:
            self.transcript.connect()
        except Exception as e:
            logger.warning(f"Discord transcript logger disabled: {e}")

        # Per-channel serialization and throttling (avoid overlapping LLM calls)
        self._channel_locks: dict[int, asyncio.Lock] = {}
        self._last_invoke_unix: dict[int, float] = {}
        self._min_invoke_interval = float(os.getenv("DISCORD_MIN_INVOKE_INTERVAL_SEC", "6.0"))

        # Optional "silent acknowledgement" reactions for no-response outcomes.
        self._no_response_reactions = self._truthy_env("DISCORD_NO_RESPONSE_REACTIONS", default="1")
        self._no_response_reactions_on_bots = self._truthy_env("DISCORD_NO_RESPONSE_REACTIONS_ON_BOTS", default="1")
        self._no_response_reaction_palette = self._parse_reaction_palette(
            os.getenv("DISCORD_NO_RESPONSE_EMOJIS", "👍,👀,✅")
        )
        self._rng = random.Random()

        # Anti-spam guardrail for high-traffic follow-mode channels.
        self._follow_window_sec = max(1, int(os.getenv("DISCORD_FOLLOW_SPAM_WINDOW_SEC", "30")))
        self._follow_max_events = max(0, int(os.getenv("DISCORD_FOLLOW_SPAM_MAX_EVENTS", "12")))
        self._follow_cooldown_sec = max(0, int(os.getenv("DISCORD_FOLLOW_SPAM_COOLDOWN_SEC", "45")))
        self._follow_recent_events: dict[int, deque[float]] = defaultdict(deque)
        self._follow_cooldown_until: dict[int, float] = {}

        # Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        # members intent is privileged; only enable if explicitly requested.
        if os.getenv("DISCORD_ENABLE_MEMBERS_INTENT", "").strip() == "1":
            intents.members = True
        
        # Create bot
        self.bot = commands.Bot(
            command_prefix="!busy38 ",
            intents=intents,
            help_command=None
        )

        # Expose bot instance for vendor async cheatcodes.
        _set_discord_runtime_bot(self.bot)

        # Forum-task-board subscriptions
        self._forums_file = Path(os.getenv("DISCORD_FORUMS_PATH", "./data/discord_forums.json"))
        self.forums: set[int] = set()
        self._seen_forum_threads: set[int] = set()
        self._load_forums()
        
        self._setup_handlers()

    @staticmethod
    def _truthy_env(name: str, *, default: str = "0") -> bool:
        raw = os.getenv(name, default).strip().lower()
        return raw not in ("", "0", "false", "no", "off")

    @staticmethod
    def _parse_reaction_palette(raw: str) -> List[str]:
        out: List[str] = []
        for part in (raw or "").split(","):
            e = part.strip()
            if e:
                out.append(e)
        return out

    @staticmethod
    def _coordination_hints(*, content: str, self_id: int, mentioned_ids: List[int], mentioned_bot_ids: List[int]) -> Dict[str, Any]:
        text = (content or "").strip().lower()
        has_assignment = bool(
            re.search(r"\b(assign|assigned|handle|take|own|owner|pick\s*up|you\s+take)\b", text)
        )
        has_handoff = bool(
            re.search(r"\b(handoff|hand\s*off|delegate|pass\s+to|route\s+to|transfer)\b", text)
        )

        mentions_self = self_id in set(mentioned_ids)
        mentioned_other_agents = [mid for mid in mentioned_bot_ids if int(mid) != int(self_id)]

        explicit_assignment_to_self = bool(mentions_self and has_assignment)
        explicit_assignment_to_other = bool((not mentions_self) and mentioned_other_agents and has_assignment)
        handoff_to_self = bool(mentions_self and has_handoff)
        handoff_to_other = bool((not mentions_self) and mentioned_other_agents and has_handoff)

        return {
            "has_assignment": has_assignment,
            "has_handoff": has_handoff,
            "mentions_self": mentions_self,
            "mentioned_other_agents": mentioned_other_agents,
            "explicit_assignment_to_self": explicit_assignment_to_self,
            "explicit_assignment_to_other": explicit_assignment_to_other,
            "handoff_to_self": handoff_to_self,
            "handoff_to_other": handoff_to_other,
        }

    def _follow_guardrail_allows(self, channel_id: int) -> bool:
        """
        Return True when follow-mode invocation is allowed for this channel.

        Strategy:
        - Keep a rolling window of recent follow-triggered events.
        - If the event count exceeds threshold, enter cooldown for the channel.
        """
        if self._follow_max_events <= 0:
            return True

        now = time.time()
        cooldown_until = float(self._follow_cooldown_until.get(channel_id, 0.0))
        if cooldown_until > now:
            return False

        q = self._follow_recent_events[channel_id]
        while q and (now - q[0]) > float(self._follow_window_sec):
            q.popleft()
        q.append(now)

        if len(q) > int(self._follow_max_events):
            self._follow_cooldown_until[channel_id] = now + float(self._follow_cooldown_sec)
            q.clear()
            return False
        return True

    @staticmethod
    def _parse_no_response_directives(result: Optional[str]) -> tuple[bool, Optional[str]]:
        """
        Parse no-response and optional reaction directives from LLM output.

        Supported:
        - [no-response /]
        - [No response required]
        - [react:🙂]
        """
        text = (result or "").strip()
        if not text:
            return False, None
        text_l = text.lower()
        is_silent = text_l == "[no response required]" or "[no-response /]" in text_l
        m = re.search(r"\[\s*react\s*:\s*([^\]\s]+)\s*\]", text, flags=re.IGNORECASE)
        reaction = m.group(1).strip() if m else None
        return is_silent, reaction

    def _load_subscriptions(self) -> None:
        """Load channel subscriptions from SquidKeys or local file."""
        data = None
        if self.keystore:
            data = load_subscriptions_from_keystore(self.keystore)
        if data is None and self._sub_file.exists():
            try:
                data = json.loads(self._sub_file.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if data:
            self.state.import_subscriptions(data)

    def _save_subscriptions(self, *, actor: str = "discord_bot") -> None:
        """Save subscriptions to SquidKeys if available, and always to local file."""
        data = self.state.export_subscriptions()
        try:
            self._sub_file.parent.mkdir(parents=True, exist_ok=True)
            self._sub_file.write_text(
                json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to write subscriptions file: {e}")
        if self.keystore:
            try:
                save_subscriptions_to_keystore(self.keystore, data, actor=actor)
            except Exception as e:
                logger.warning(f"Failed to save subscriptions to SquidKeys: {e}")

    def _get_forums_from_squidkeys(self) -> Optional[set]:
        if not self.keystore:
            return None
        try:
            record = self.keystore.get_password(
                agent_id="busy38",
                name="discord_forums",
                actor="discord_bot",
            )
            if not record:
                return None
            forum_ids = json.loads(record.password)
            return set(int(x) for x in forum_ids)
        except Exception:
            return None

    def _parse_forums_env(self) -> Optional[set]:
        raw = os.getenv("DISCORD_FORUMS")
        if not raw:
            return None
        out = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.add(int(part))
            except Exception:
                continue
        return out or None

    def _load_forums(self) -> None:
        data = self._get_forums_from_squidkeys() or self._parse_forums_env()
        if data is None and self._forums_file.exists():
            try:
                data = set(int(x) for x in json.loads(self._forums_file.read_text(encoding="utf-8")))
            except Exception:
                data = None
        self.forums = set(data or [])

    def _save_forums(self, *, actor: str = "discord_bot") -> None:
        try:
            self._forums_file.parent.mkdir(parents=True, exist_ok=True)
            self._forums_file.write_text(
                json.dumps(sorted(list(self.forums)), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to write forums file: {e}")
        if self.keystore:
            try:
                self.keystore.save_password(
                    agent_id="busy38",
                    name="discord_forums",
                    password=json.dumps(sorted(list(self.forums))),
                    metadata={"count": len(self.forums)},
                    actor=actor,
                )
            except Exception as e:
                logger.warning(f"Failed to save forums to SquidKeys: {e}")

    def _lock_for_channel(self, channel_id: int) -> asyncio.Lock:
        lock = self._channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_id] = lock
        return lock

    def _should_invoke_agent(self, *, key: ChannelKey, is_dm: bool, is_mentioned: bool, is_reply_to_me: bool, content: str) -> bool:
        """
        Decide whether to call the LLM.

        We always ingest/log messages for context, but only invoke on:
        - DMs (authorized)
        - explicit mention / reply-to-me
        - wakeword prefix
        - subscribed follow-mode (throttled)
        """
        content_l = (content or "").strip().lower()
        wakeword = content_l.startswith("busy38:") or content_l.startswith("squidder:")
        cfg = self.state.channel_config(key)

        if is_dm:
            return True
        if is_mentioned or is_reply_to_me or wakeword:
            return True
        if cfg.subscribed and cfg.follow_mode:
            # Follow-mode means "full citizen": observe everything, respond when relevant.
            # Still throttle to avoid spamming LLM calls.
            now = time.time()
            last = self._last_invoke_unix.get(key.channel_id, 0.0)
            if now - last >= self._min_invoke_interval:
                return True
        return False

    def _build_discord_system_prompt(self, *, key: ChannelKey, channel_name: str, guild_name: Optional[str]) -> str:
        cfg = self.state.channel_config(key)
        agents = sorted(list(self.state.known_bot_agents(key.guild_id)))
        agents_str = ", ".join(str(a) for a in agents) if agents else "none observed"

        return (
            "You are Busy38 running as a Discord bot.\n\n"
            "You can see recent channel history and should behave like a normal participant.\n"
            "If no response is needed, output exactly: [no-response /]\n"
            "Optional: include [react:EMOJI] to react to the triggering message when staying silent.\n\n"
            f"Context:\n"
            f"- guild: {guild_name or 'DM'}\n"
            f"- channel: {channel_name} (id={key.channel_id})\n"
            f"- subscribed: {cfg.subscribed}\n"
            f"- follow_mode: {cfg.follow_mode}\n"
            f"- other bot agents observed in this guild: {agents_str}\n\n"
            "Discord constraints:\n"
            "- Keep messages under ~1800 chars when possible.\n"
            "- If you need multiple parts, label them (part 1/2, 2/2).\n\n"
            "Multi-agent coordination:\n"
            "- If a task is clearly assigned or handed off to another agent, prefer [no-response /].\n"
            "- If assigned/handed off to you, acknowledge briefly and take ownership.\n"
            "- Keep handoff replies explicit and tag the next owner when delegating.\n\n"
            "Busy38 tool use:\n"
            "Use CHEATCODES for operations and [next /] for multi-step work.\n"
            "Discord task boards (forums):\n"
            "- Use dforum cheatcodes to post updates and manage forum-thread state when needed.\n"
        )

    def _build_context_messages(self, *, key: ChannelKey, system_prompt: str, user_task: str) -> List[Dict]:
        # Default recency bias: last 24h.
        max_age_sec = int(os.getenv("DISCORD_CONTEXT_MAX_AGE_SEC", str(24 * 60 * 60)))
        lines = self.state.prompt_context_lines(
            key,
            max_messages=int(os.getenv("DISCORD_CONTEXT_MESSAGES", "40")),
            max_age_sec=None if max_age_sec <= 0 else max_age_sec,
        )
        history_block = "\n".join(lines) if lines else "(no history available)"
        return [
            {"role": "system", "content": system_prompt + "\nRecent channel history:\n" + history_block},
            {"role": "user", "content": user_task},
        ]
    
    def _get_token_from_squidkeys(self) -> Optional[str]:
        """Fetch Discord token from SquidKeys."""
        if not self.keystore:
            return None
        try:
            record = self.keystore.get_authorization(
                agent_id="busy38",
                provider="discord",
                actor="discord_bot"
            )
            if record:
                logger.info("Retrieved Discord token from SquidKeys")
                return record.access_token
        except Exception as e:
            logger.debug(f"Could not retrieve token from SquidKeys: {e}")
        return None
    
    def _get_allowed_users_from_squidkeys(self) -> Optional[set]:
        """Fetch allowed users list from SquidKeys."""
        if not self.keystore:
            return None
        try:
            record = self.keystore.get_password(
                agent_id="busy38",
                name="discord_allowed_users",
                actor="discord_bot"
            )
            if record:
                # Parse JSON list from password field
                users = json.loads(record.password)
                logger.info(f"Retrieved {len(users)} allowed users from SquidKeys")
                return set(int(uid) for uid in users)
        except Exception as e:
            logger.debug(f"Could not retrieve allowed users from SquidKeys: {e}")
        return None
    
    def _parse_allowed_users_env(self) -> Optional[set]:
        """Parse ALLOWED_USERS environment variable."""
        users_str = os.getenv("ALLOWED_USERS")
        if not users_str:
            return None
        try:
            users = [int(uid.strip()) for uid in users_str.split(",")]
            logger.info(f"Using {len(users)} allowed users from ALLOWED_USERS env var")
            return set(users)
        except ValueError:
            logger.warning("ALLOWED_USERS env var is not valid comma-separated integers")
            return None
    
    def _setup_handlers(self):
        """Set up Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            """Called when bot connects to Discord."""
            logger.info(f"Discord bot logged in as {self.bot.user}")
            print(f"✓ Discord bot connected: {self.bot.user}")
            print(f"  Guilds: {len(self.bot.guilds)}")
            for guild in self.bot.guilds:
                print(f"    - {guild.name} ({guild.id})")

            # Warm state with history for subscribed channels (best-effort).
            await self._warm_history()

        async def _safe_channel_history(channel, limit: int):
            try:
                async for msg in channel.history(limit=limit, oldest_first=True):
                    yield msg
            except Exception:
                return

        async def _channel_display_name(channel) -> str:
            try:
                return getattr(channel, "name", None) or str(channel)
            except Exception:
                return "unknown"

        async def _guild_name(guild) -> Optional[str]:
            try:
                return guild.name if guild else None
            except Exception:
                return None

        async def _ingest_message(message) -> None:
            # Build key
            gid = message.guild.id if getattr(message, "guild", None) else None
            key = ChannelKey(guild_id=gid, channel_id=message.channel.id)

            # Normalize content (include attachments as URLs)
            content = message.content or ""
            try:
                atts = getattr(message, "attachments", None) or []
                if atts:
                    urls = [a.url for a in atts if getattr(a, "url", None)]
                    if urls:
                        content = (content + "\n" + "\n".join(urls)).strip()
            except Exception:
                pass

            rec = MessageRecord(
                message_id=message.id,
                created_at_unix=message.created_at.timestamp() if getattr(message, "created_at", None) else time.time(),
                author_id=message.author.id,
                author_name=str(message.author),
                content=content,
                is_bot=bool(getattr(message.author, "bot", False)),
                reply_to_id=getattr(getattr(message, "reference", None), "message_id", None),
            )
            self.state.ingest_message(key, rec)

            # Log transcript for "boards"/analysis.
            try:
                proj = f"discord:{gid}:{message.channel.id}"
                self.transcript.log_message(
                    source_id=f"discord:{message.id}",
                    timestamp=message.created_at,
                    content=content,
                    project_id=proj,
                    participants=[message.author.id],
                    topics=[],
                    metadata={
                        "guild_id": gid,
                        "channel_id": message.channel.id,
                        "message_id": message.id,
                        "author_id": message.author.id,
                        "author_name": str(message.author),
                        "is_bot": bool(getattr(message.author, "bot", False)),
                    },
                )
            except Exception:
                pass

        async def _is_reply_to_me(message) -> bool:
            try:
                ref = getattr(message, "reference", None)
                if not ref or not ref.message_id:
                    return False
                # This may fetch if not cached; keep best-effort.
                ref_msg = ref.resolved
                if ref_msg is None:
                    try:
                        ref_msg = await message.channel.fetch_message(ref.message_id)
                    except Exception:
                        return False
                return bool(ref_msg and ref_msg.author and ref_msg.author.id == self.bot.user.id)
            except Exception:
                return False

        async def _is_command_message(message) -> bool:
            try:
                return message.content.startswith("!busy38 ")
            except Exception:
                return False

        async def _maybe_handle_command(message) -> bool:
            # Let commands.Bot handle registered commands.
            if await _is_command_message(message):
                await self.bot.process_commands(message)
                return True
            return False

        async def _invoke_agent_for_message(
            message,
            content: str,
            *,
            is_dm: bool,
            is_mentioned: bool,
            trigger_override: Optional[str] = None,
            coordination: Optional[Dict[str, Any]] = None,
        ) -> Optional[str]:
            gid = message.guild.id if getattr(message, "guild", None) else None
            key = ChannelKey(guild_id=gid, channel_id=message.channel.id)
            channel_name = await _channel_display_name(message.channel)
            guild_name = await _guild_name(message.guild if getattr(message, "guild", None) else None)
            sys_prompt = self._build_discord_system_prompt(key=key, channel_name=channel_name, guild_name=guild_name)
            coord = coordination or {}
            mentioned_other_agents = coord.get("mentioned_other_agents") or []

            # If not explicitly mentioned, give agent a strong prior to be silent unless needed.
            trigger = trigger_override or ("mention" if is_mentioned else ("dm" if is_dm else "follow"))
            user_task = (
                f"Trigger: {trigger}\n"
                f"Coordination: assignment={bool(coord.get('has_assignment'))}, handoff={bool(coord.get('has_handoff'))}, "
                f"mentions_self={bool(coord.get('mentions_self'))}, "
                f"assigned_to_self={bool(coord.get('explicit_assignment_to_self'))}, "
                f"assigned_to_other={bool(coord.get('explicit_assignment_to_other'))}, "
                f"handoff_to_self={bool(coord.get('handoff_to_self'))}, "
                f"handoff_to_other={bool(coord.get('handoff_to_other'))}, "
                f"other_agent_ids={','.join(str(x) for x in mentioned_other_agents) or 'none'}\n"
                f"New message from {message.author} ({message.author.id}):\n"
                f"{content}\n\n"
                "Respond as Busy38 in this Discord channel, or output [no-response /] if you should stay silent."
            )

            ctx_messages = self._build_context_messages(key=key, system_prompt=sys_prompt, user_task=user_task)
            return await self.orchestrator.run_agent_loop(task=user_task, context=ctx_messages)
        
        @self.bot.event
        async def on_message(message):
            """Handle incoming messages.
            
            Agent coordination mode: Bot sees ALL messages in channel
            and decides when to respond using [no-response /] if not needed.
            """
            # Ignore own messages
            if message.author == self.bot.user:
                return
            
            # Note: We process ALL messages from humans and other bots
            # for inter-agent coordination. The agent decides when to respond.
            
            # Only skip if it's from us
            if message.author.id == self.bot.user.id:
                return

            # Let command router run first (subscribe/follow/etc).
            if await _maybe_handle_command(message):
                return
            
            # Check if this is a DM (for authorization check)
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            # Check if we're specifically mentioned (for priority handling)
            is_mentioned = self.bot.user in message.mentions
            bot_mention_str = f"<@{self.bot.user.id}>"
            bot_mention_str2 = f"<@!{self.bot.user.id}>"
            is_mentioned = is_mentioned or bot_mention_str in message.content or bot_mention_str2 in message.content
            mention_ids = [int(m.id) for m in (getattr(message, "mentions", None) or [])]
            mention_bot_ids = [
                int(m.id) for m in (getattr(message, "mentions", None) or []) if bool(getattr(m, "bot", False))
            ]

            is_reply_to_me = await _is_reply_to_me(message)
            
            # DM Authorization check
            if is_dm and self.allowed_users is not None:
                if message.author.id not in self.allowed_users:
                    logger.warning(f"Unauthorized DM from {message.author} ({message.author.id})")
                    await message.channel.send(
                        "🔒 Sorry, you're not authorized to DM this bot.\n"
                        "Please contact an admin to get access."
                    )
                    return

            # Ingest message into state + transcript (authorized DMs + all channel traffic).
            await _ingest_message(message)
            
            # Remove bot mention from message
            content = message.content
            if is_mentioned:
                content = content.replace(f"<@{self.bot.user.id}>", "").strip()
                content = content.replace(f"<@!{self.bot.user.id}>", "").strip()

            # Wakeword prefixes (treat like an implicit mention)
            lowered = (content or "").lstrip().lower()
            if lowered.startswith("busy38:"):
                content = content.split(":", 1)[1].strip()
                is_mentioned = True
            elif lowered.startswith("squidder:"):
                content = content.split(":", 1)[1].strip()
                is_mentioned = True

            coordination = self._coordination_hints(
                content=content,
                self_id=self.bot.user.id,
                mentioned_ids=mention_ids,
                mentioned_bot_ids=mention_bot_ids,
            )

            # Forum task board: detect threads under subscribed forum channels.
            is_forum_thread = False
            is_new_forum_task = False
            try:
                if isinstance(message.channel, discord.Thread) and getattr(message.channel, "parent", None) is not None:
                    parent = message.channel.parent
                    # Forum channels are a task-board surface.
                    if isinstance(parent, discord.ForumChannel) and parent.id in self.forums:
                        is_forum_thread = True
                        if message.channel.id not in self._seen_forum_threads and not bool(getattr(message.author, "bot", False)):
                            is_new_forum_task = True
                            self._seen_forum_threads.add(message.channel.id)
                            try:
                                await message.channel.join()
                            except Exception:
                                pass
            except Exception:
                pass
            
            if not content:
                await message.channel.send("👋 Hello! I'm Busy38. How can I help?")
                return
            
            logger.info(f"Discord message from {message.author}: {content[:50]}...")

            gid = message.guild.id if getattr(message, "guild", None) else None
            key = ChannelKey(guild_id=gid, channel_id=message.channel.id)
            should_invoke = self._should_invoke_agent(
                key=key,
                is_dm=is_dm,
                is_mentioned=is_mentioned,
                is_reply_to_me=is_reply_to_me,
                content=content,
            )
            if is_new_forum_task:
                should_invoke = True
            if not should_invoke:
                return

            # If the message explicitly assigns/hand-offs to another agent, stay quiet by default.
            if (
                not is_dm
                and not is_mentioned
                and not is_reply_to_me
                and (coordination.get("explicit_assignment_to_other") or coordination.get("handoff_to_other"))
            ):
                return

            cfg = self.state.channel_config(key)
            is_follow_trigger = (
                (not is_dm)
                and (not is_mentioned)
                and (not is_reply_to_me)
                and (not is_new_forum_task)
                and bool(cfg.subscribed and cfg.follow_mode)
            )
            if is_follow_trigger and (not self._follow_guardrail_allows(message.channel.id)):
                return

            # Serialize per-channel to avoid overlapping LLM calls.
            lock = self._lock_for_channel(message.channel.id)
            if lock.locked() and not (is_dm or is_mentioned or is_reply_to_me):
                # If we're already responding, don't queue follow chatter.
                return
            
            # Show typing indicator
            async with message.channel.typing():
                try:
                    async with lock:
                        self._last_invoke_unix[message.channel.id] = time.time()

                        # Route to orchestrator with full channel context.
                        result = await _invoke_agent_for_message(
                            message,
                            content,
                            is_dm=is_dm,
                            is_mentioned=is_mentioned or is_reply_to_me,
                            trigger_override="forum_task" if is_new_forum_task else None,
                            coordination=coordination,
                        )

                    # True silence: the agent can emit [no-response /], optional [react:EMOJI].
                    is_silent, requested_reaction = self._parse_no_response_directives(result)
                    if is_silent:
                        if self._no_response_reactions:
                            if (not bool(getattr(message.author, "bot", False))) or self._no_response_reactions_on_bots:
                                emoji = requested_reaction
                                if not emoji and self._no_response_reaction_palette:
                                    emoji = self._rng.choice(self._no_response_reaction_palette)
                                if emoji:
                                    try:
                                        await message.add_reaction(emoji)
                                    except Exception:
                                        pass
                        return
                    
                    # Send response (split if too long)
                    if len(result) > 2000:
                        # Discord limit is 2000 chars
                        chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                        for i, chunk in enumerate(chunks):
                            suffix = f"\n\n(part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                            await message.channel.send(chunk + suffix)
                    else:
                        await message.channel.send(result)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await message.channel.send(f"❌ Error: {str(e)[:500]}")
        
        # Simple commands
        @self.bot.command(name="status")
        async def status_cmd(ctx):
            """Show Busy38 status."""
            status = self.orchestrator.get_status()
            await ctx.send(
                f"**Busy38 Status**\n"
                f"Running: {status['running']}\n"
                f"Namespaces: {', '.join(status['namespaces'])}\n"
                f"Iteration: {status.get('iteration', 0)}"
            )

        @self.bot.command(name="subscribe")
        @commands.is_owner()
        async def subscribe_cmd(ctx):
            """Subscribe current channel for context tracking (and optional follow-mode)."""
            gid = ctx.guild.id if ctx.guild else None
            key = ChannelKey(guild_id=gid, channel_id=ctx.channel.id)
            self.state.set_subscribed(key, True)
            self._save_subscriptions(actor=str(ctx.author.id))
            await ctx.send(f"✓ Subscribed this channel (follow_mode={self.state.channel_config(key).follow_mode})")

        @self.bot.command(name="unsubscribe")
        @commands.is_owner()
        async def unsubscribe_cmd(ctx):
            """Unsubscribe current channel."""
            gid = ctx.guild.id if ctx.guild else None
            key = ChannelKey(guild_id=gid, channel_id=ctx.channel.id)
            self.state.set_subscribed(key, False)
            self.state.set_follow_mode(key, False)
            self._save_subscriptions(actor=str(ctx.author.id))
            await ctx.send("✓ Unsubscribed this channel")

        @self.bot.command(name="follow")
        @commands.is_owner()
        async def follow_cmd(ctx, mode: str = "on"):
            """Toggle follow-mode for the current channel (on/off)."""
            gid = ctx.guild.id if ctx.guild else None
            key = ChannelKey(guild_id=gid, channel_id=ctx.channel.id)
            m = mode.strip().lower()
            if m not in ("on", "off"):
                await ctx.send("Usage: `!busy38 follow on` or `!busy38 follow off`")
                return
            self.state.set_subscribed(key, True)
            self.state.set_follow_mode(key, m == "on")
            self._save_subscriptions(actor=str(ctx.author.id))
            await ctx.send(f"✓ follow_mode={m} (subscribed=True)")

        @self.bot.command(name="subs")
        @commands.is_owner()
        async def subs_cmd(ctx):
            """List subscribed channels."""
            subs = self.state.export_subscriptions()
            if not subs:
                await ctx.send("No subscribed channels.")
                return
            lines = ["Subscribed channels:"]
            for k, v in sorted(subs.items()):
                lines.append(f"- `{k}` follow_mode={v.get('follow_mode')} history_limit={v.get('history_limit')}")
            await ctx.send("\n".join(lines))

        @self.bot.command(name="forum_subscribe")
        @commands.is_owner()
        async def forum_subscribe_cmd(ctx):
            """Subscribe a forum channel as a task board (run inside the forum channel)."""
            if isinstance(ctx.channel, discord.Thread) and getattr(ctx.channel, "parent", None) is not None:
                ch = ctx.channel.parent
            else:
                ch = ctx.channel
            if not isinstance(ch, discord.ForumChannel):
                await ctx.send("Run this in a forum channel (or in a thread under that forum).")
                return
            self.forums.add(ch.id)
            self._save_forums(actor=str(ctx.author.id))
            await ctx.send(f"✓ Subscribed forum `{ch.name}` ({ch.id})")

        @self.bot.command(name="forum_unsubscribe")
        @commands.is_owner()
        async def forum_unsubscribe_cmd(ctx):
            """Unsubscribe a forum channel (run inside the forum channel)."""
            if isinstance(ctx.channel, discord.Thread) and getattr(ctx.channel, "parent", None) is not None:
                ch = ctx.channel.parent
            else:
                ch = ctx.channel
            if not isinstance(ch, discord.ForumChannel):
                await ctx.send("Run this in a forum channel (or in a thread under that forum).")
                return
            self.forums.discard(ch.id)
            self._save_forums(actor=str(ctx.author.id))
            await ctx.send(f"✓ Unsubscribed forum `{ch.name}` ({ch.id})")

        @self.bot.command(name="forums")
        @commands.is_owner()
        async def forums_cmd(ctx):
            """List subscribed forums."""
            if not self.forums:
                await ctx.send("No subscribed forums.")
                return
            lines = ["Subscribed forums:"]
            for fid in sorted(self.forums):
                lines.append(f"- `{fid}`")
            await ctx.send("\n".join(lines))

        @self.bot.command(name="agents")
        async def agents_cmd(ctx):
            """List other bot agents observed in this guild."""
            if not ctx.guild:
                await ctx.send("DM has no guild agent roster.")
                return
            ids = sorted(list(self.state.known_bot_agents(ctx.guild.id)))
            if not ids:
                await ctx.send("No other bot agents observed yet.")
                return
            await ctx.send("Observed bot agents:\n" + "\n".join([f"- <@{i}>" for i in ids]))

        @self.bot.command(name="fingerprint")
        async def fingerprint_cmd(ctx):
            """Show this machine fingerprint (for hardware auth provisioning)."""
            from core.security.hardware_auth import machine_fingerprint
            await ctx.send(f"`{machine_fingerprint()}`")

        @self.bot.command(name="auth")
        async def auth_cmd(ctx):
            """Show DM authorization status."""
            squidkeys_status = "✓ Connected" if self.keystore else "✗ Not available"
            
            if self.allowed_users is None:
                await ctx.send(
                    f"🔓 DMs are open to all users\n"
                    f"SquidKeys: {squidkeys_status}\n\n"
                    f"Your ID: {ctx.author.id}"
                )
            else:
                users_str = "\n".join([f"- <@{uid}>" for uid in self.allowed_users])
                await ctx.send(
                    f"🔒 DMs restricted to {len(self.allowed_users)} user(s):\n{users_str}\n\n"
                    f"SquidKeys: {squidkeys_status}\n"
                    f"Your ID: {ctx.author.id}"
                )
        
        @self.bot.command(name="save_token")
        @commands.is_owner()
        async def save_token_cmd(ctx, token: str):
            """Save Discord token to SquidKeys (owner only)."""
            if not self.keystore:
                await ctx.send("❌ SquidKeys not available")
                return
            
            try:
                self.keystore.save_authorization(
                    agent_id="busy38",
                    provider="discord",
                    access_token=token,
                    metadata={"saved_by": str(ctx.author.id), "source": "discord_command"},
                    actor=str(ctx.author.id)
                )
                await ctx.send("✓ Token saved to SquidKeys. Restart bot to use it.")
            except Exception as e:
                await ctx.send(f"❌ Failed to save token: {e}")
        
        @self.bot.command(name="save_users")
        @commands.is_owner()
        async def save_users_cmd(ctx, *, users: str):
            """Save allowed users list to SquidKeys (owner only).
            
            Usage: !busy38 save_users 123456789,987654321
            """
            if not self.keystore:
                await ctx.send("❌ SquidKeys not available")
                return
            
            try:
                user_ids = [int(uid.strip()) for uid in users.split(",")]
                self.keystore.save_password(
                    agent_id="busy38",
                    name="discord_allowed_users",
                    password=json.dumps(user_ids),
                    metadata={"saved_by": str(ctx.author.id), "count": len(user_ids)},
                    actor=str(ctx.author.id)
                )
                await ctx.send(f"✓ {len(user_ids)} user(s) saved to SquidKeys. Restart bot to apply.")
            except ValueError:
                await ctx.send("❌ Invalid format. Use comma-separated user IDs: `123,456`")
            except Exception as e:
                await ctx.send(f"❌ Failed to save users: {e}")
        
        @self.bot.command(name="help")
        async def help_cmd(ctx):
            """Show help."""
            auth_status = "🔒 Restricted" if self.allowed_users else "🔓 Open"
            squidkeys_status = "✓" if self.keystore else "✗"
            
            help_text = (
                "**Busy38 Discord Bot**\n\n"
                "Mention me, reply to me, DM me, or use `busy38:` / `squidder:` to chat.\n\n"
                "Commands:\n"
                "`!busy38 status` - Show system status\n"
                "`!busy38 auth` - Show DM authorization status\n"
                "`!busy38 agents` - List observed bot agents\n"
                "`!busy38 help` - Show this help\n\n"
                f"DM Access: {auth_status}\n"
                f"SquidKeys: {squidkeys_status}\n\n"
            )
            
            # Add owner-only commands if user is owner
            if await self.bot.is_owner(ctx.author):
                help_text += (
                    "Owner Commands:\n"
                    "`!busy38 save_token <token>` - Save Discord token to SquidKeys\n"
                    "`!busy38 save_users <ids>` - Save allowed users (comma-separated)\n\n"
                    "`!busy38 subscribe` - Subscribe current channel\n"
                    "`!busy38 unsubscribe` - Unsubscribe current channel\n"
                    "`!busy38 follow on|off` - Toggle follow-mode for current channel\n"
                    "`!busy38 subs` - List subscriptions\n\n"
                    "`!busy38 forum_subscribe` - Subscribe this forum as a task board\n"
                    "`!busy38 forum_unsubscribe` - Unsubscribe this forum\n"
                    "`!busy38 forums` - List forum subscriptions\n\n"
                )
            
            help_text += (
                "I understand cheatcodes like:\n"
                "`[rw4:read_file path=\"...\" /]`\n"
                "`[rw4:list path=\"...\" /]`\n"
                "`[next /]` - Continue after tool use\n"
                "`[no-response /]` - Silent response"
            )
            
            await ctx.send(help_text)
    
    async def _warm_history(self) -> None:
        """Fetch recent history for subscribed channels (best-effort)."""
        subs = self.state.list_subscriptions()
        if not subs:
            return
        limit = int(os.getenv("DISCORD_WARM_HISTORY_LIMIT", "50"))

        for key, cfg in subs:
            if key.guild_id is None:
                continue
            guild = self.bot.get_guild(key.guild_id)
            if not guild:
                continue
            channel = guild.get_channel(key.channel_id)
            if not channel:
                continue
            try:
                async for msg in channel.history(limit=limit, oldest_first=True):
                    # Reuse on_message ingestion path.
                    try:
                        gid = msg.guild.id if getattr(msg, "guild", None) else None
                        k2 = ChannelKey(guild_id=gid, channel_id=msg.channel.id)
                        content = msg.content or ""
                        rec = MessageRecord(
                            message_id=msg.id,
                            created_at_unix=msg.created_at.timestamp() if getattr(msg, "created_at", None) else time.time(),
                            author_id=msg.author.id,
                            author_name=str(msg.author),
                            content=content,
                            is_bot=bool(getattr(msg.author, "bot", False)),
                            reply_to_id=getattr(getattr(msg, "reference", None), "message_id", None),
                        )
                        self.state.ingest_message(k2, rec)
                    except Exception:
                        continue
            except Exception:
                continue

    async def start(self):
        """Start the Discord bot."""
        # Ensure core cheatcode namespaces are registered (rw4, etc.).
        import core.cheatcodes.setup  # noqa: F401

        # Load vendor plugins so agent-internal tools (like dlog) are available.
        try:
            pm = PluginManager(vendor_path="./vendor")
            await pm.load_plugins()
        except Exception as e:
            logger.warning(f"Vendor plugin load skipped/failed: {e}")

        # Hardware-bound auth gate (opt-in unless configured).
        # Enable explicitly with BUSY38_REQUIRE_HARDWARE_AUTH=1, or by providing both license+pubkey.
        enforce = os.getenv("BUSY38_REQUIRE_HARDWARE_AUTH", "").strip() == "1" or hardware_auth_is_configured()
        if enforce:
            try:
                lic = require_hardware_auth()
                logger.info(f"Hardware auth OK (license={lic.version})")
            except HardwareAuthError as e:
                # Fail closed when enforced.
                raise RuntimeError(str(e)) from e
        else:
            logger.warning(
                "Hardware auth is not enforced (no license configured). "
                "Set BUSY38_REQUIRE_HARDWARE_AUTH=1 and configure BUSY38_LICENSE_* to enable."
            )

        if self._owns_orchestrator:
            await self.orchestrator.start()
        
        logger.info("Starting Discord bot...")
        await self.bot.start(self.token)
    
    async def stop(self):
        """Stop the Discord bot."""
        logger.info("Stopping Discord bot...")
        await self.bot.close()

        try:
            self.transcript.close()
        except Exception:
            pass
        
        if self._owns_orchestrator:
            await self.orchestrator.stop()
    
    def run(self):
        """Run the bot (blocking)."""
        asyncio.run(self.start())


async def main():
    """Main entry point for Discord bot."""
    # Check for token
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("Error: DISCORD_TOKEN environment variable not set")
        print("Get your bot token from: https://discord.com/developers/applications")
        return 1
    
    # Create and start bot
    bot = Busy38DiscordBot(token=token)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await bot.stop()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
