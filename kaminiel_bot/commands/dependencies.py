from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, MutableMapping, MutableSet, Optional, Sequence, Tuple, Union

import discord


@dataclass(slots=True)
class HeartbeatDependencies:
    preferences: MutableMapping[int, str]
    save_preferences: Callable[[], None]
    describe_preference: Callable[[discord.Guild], str]
    is_creator_user: Callable[[Optional[discord.abc.User]], bool]
    can_manage_guild: Callable[[discord.Guild, discord.abc.User], bool]


@dataclass(slots=True)
class NudgeDependencies:
    server_preferences: MutableMapping[int, MutableMapping[str, Any]]
    delivery_preferences: MutableMapping[int, str]
    dm_assignments: MutableMapping[int, MutableMapping[str, Any]]
    save_preferences: Callable[[], None]
    describe_preference: Callable[[discord.Guild], str]
    sanitize_display_name: Callable[[Any], str]
    is_creator_user: Callable[[Optional[discord.abc.User]], bool]
    can_manage_guild: Callable[[discord.Guild, discord.abc.User], bool]
    default_delivery_mode: str


@dataclass(slots=True)
class AnnounceDependencies:
    cache: MutableMapping[int, MutableMapping[str, Any]]
    save_cache: Callable[[], None]
    auto_select_channel: Callable[[discord.Guild, bool], Optional[discord.TextChannel]]
    resolve_guild_for_user: Callable[[discord.abc.User, Optional[str]], Tuple[Optional[discord.Guild], Optional[str]]]
    mutual_guilds_for_user: Callable[[discord.abc.User], Sequence[discord.Guild]]
    bot_can_speak: Callable[[discord.abc.GuildChannel], bool]
    sanitize_display_name: Callable[[Any], str]
    is_creator_user: Callable[[Optional[discord.abc.User]], bool]
    can_manage_guild: Callable[[discord.Guild, discord.abc.User], bool]
    personalization: MutableMapping[int, str]
    save_personalization: Callable[[], None]
    describe_personalization: Callable[[discord.Guild], str]
    default_personalization: str


@dataclass(slots=True)
class VoiceDependencies:
    channel_cache: MutableMapping[int, int]
    save_cache: Callable[[], None]
    get_voice_lock: Callable[[int], asyncio.Lock]
    cancel_voice_disconnect: Callable[[int], None]
    creator_user_ids: Sequence[int]
    voice_flags: MutableMapping[str, Any]
    dependency_warning_message: str
    disabled_guilds: MutableSet[int]
    save_disabled: Callable[[], None]


@dataclass(slots=True)
class ChatDependencies:
    handler: Callable[[discord.Interaction, str], Awaitable[None]]


@dataclass(slots=True)
class HelpDependencies:
    generate_help_lines: Callable[[], Sequence[str]]


@dataclass(slots=True)
class CommandDependencies:
    heartbeat: HeartbeatDependencies
    nudge: NudgeDependencies
    announce: AnnounceDependencies
    voice: VoiceDependencies
    chat: ChatDependencies
    help: HelpDependencies
