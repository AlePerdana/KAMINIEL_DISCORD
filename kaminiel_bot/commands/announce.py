from __future__ import annotations

from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import AnnounceDependencies


class AnnounceCog(commands.Cog):
    """Slash commands for managing Kaminiel's announcement destinations."""

    announce = app_commands.Group(name="announce", description="Manage where Kaminiel posts wake and heartbeat messages")
    personalize = app_commands.Group(
        name="personalize",
        description="Adjust how Kaminiel addresses announcement and heartbeat messages",
        parent=announce,
    )

    def __init__(self, bot: commands.Bot, deps: AnnounceDependencies) -> None:
        self.bot = bot
        self.deps = deps

    @announce.command(name="show", description="Display the current announcement destination")
    @app_commands.describe(guild_hint="Server name or ID (only needed when using this command in DMs)")
    async def announce_show(self, interaction: discord.Interaction, guild_hint: Optional[str] = None) -> None:
        guild = interaction.guild
        if guild is None:
            guild, error = self.deps.resolve_guild_for_user(interaction.user, guild_hint)
            if guild is None:
                await interaction.response.send_message(error or "I couldn't find that server.", ephemeral=True)
                return
        destination = self._describe_destination(guild, interaction.user)
        personalization = self.deps.describe_personalization(guild)
        message = f"{destination}\n\n{personalization}"
        await interaction.response.send_message(message, ephemeral=True)

    @announce.command(name="set_channel", description="Send announcements to a specific channel")
    @app_commands.describe(channel="Channel where Kaminiel should post announcements")
    @app_commands.guild_only()
    async def announce_set_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change the announcement channel.",
                ephemeral=True,
            )
            return

        if not self.deps.bot_can_speak(channel):
            await interaction.response.send_message(
                "I can't speak in that channel—please choose one I can send messages to.",
                ephemeral=True,
            )
            return

        self.deps.cache[guild.id] = {
            "mode": "channel",
            "channel_id": channel.id,
        }
        self.deps.save_cache()
        await interaction.response.send_message(
            f"Okay, I'll use {channel.mention} for wake, heartbeat, and farewell messages.",
            ephemeral=True,
        )

    @announce.command(name="set_dm", description="Send announcements to a DM")
    @app_commands.describe(
        user="User to receive announcements (defaults to you)",
        guild_hint="Server name or ID when using the command in DMs",
    )
    async def announce_set_dm(
        self,
        interaction: discord.Interaction,
        user: Optional[discord.User] = None,
        guild_hint: Optional[str] = None,
    ) -> None:
        guild = interaction.guild
        target_user = user or interaction.user

        if guild is None:
            guild, error = self.deps.resolve_guild_for_user(interaction.user, guild_hint)
            if guild is None:
                await interaction.response.send_message(error or "I couldn't find that server.", ephemeral=True)
                return
            member = guild.get_member(interaction.user.id)
            if member is None:
                await interaction.response.send_message(
                    "I can't find you in that server. Join it first, then try again!",
                    ephemeral=True,
                )
                return
            if target_user.id != interaction.user.id:
                await interaction.response.send_message(
                    "From DMs you can only set announcements to yourself. Hop into the server if you want to pick someone else.",
                    ephemeral=True,
                )
                return
            # Permission check is skipped in DM context; assume the user configures their own DMs only.
        else:
            if not self._can_manage(interaction):
                await interaction.response.send_message(
                    "You need Manage Server permissions (or be Kamitchi) to change the announcement channel.",
                    ephemeral=True,
                )
                return

        try:
            dm_channel = target_user.dm_channel or await target_user.create_dm()
        except discord.Forbidden:
            await interaction.response.send_message(
                "I can't open a DM with them—maybe their privacy settings block me.",
                ephemeral=True,
            )
            return
        except discord.HTTPException:
            await interaction.response.send_message(
                "I couldn't open a DM right now—please try again a little later.",
                ephemeral=True,
            )
            return

        if dm_channel is None:
            await interaction.response.send_message("I couldn't find a way to DM them.", ephemeral=True)
            return

        self.deps.cache[guild.id] = {
            "mode": "dm",
            "user_id": target_user.id,
        }
        self.deps.save_cache()

        if target_user.id == interaction.user.id:
            note = "Okay, I'll send wake/heartbeat/farewell messages straight to your DMs."
        else:
            mention = getattr(target_user, "mention", self.deps.sanitize_display_name(getattr(target_user, "name", "friend")))
            note = f"Okay, I'll send wake/heartbeat/farewell messages directly to {mention}."

        await interaction.response.send_message(note, ephemeral=True)

    @announce.command(name="clear", description="Reset announcement destination to automatic selection")
    @app_commands.describe(guild_hint="Server name or ID (only needed when using this command in DMs)")
    async def announce_clear(self, interaction: discord.Interaction, guild_hint: Optional[str] = None) -> None:
        guild = interaction.guild
        if guild is None:
            guild, error = self.deps.resolve_guild_for_user(interaction.user, guild_hint)
            if guild is None:
                await interaction.response.send_message(error or "I couldn't find that server.", ephemeral=True)
                return
        elif not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change the announcement channel.",
                ephemeral=True,
            )
            return

        if guild.id in self.deps.cache:
            self.deps.cache.pop(guild.id, None)
            self.deps.save_cache()
            message = "Announcement destination cleared—I'll pick a sensible spot next time."
        else:
            message = "No announcement destination is set for that server."

        await interaction.response.send_message(message, ephemeral=True)

    @personalize.command(name="show", description="Show Kaminiel's current announcement personalization")
    @app_commands.describe(guild_hint="Server name or ID (only needed when using this command in DMs)")
    async def personalize_show(self, interaction: discord.Interaction, guild_hint: Optional[str] = None) -> None:
        guild = interaction.guild
        if guild is None:
            guild, error = self.deps.resolve_guild_for_user(interaction.user, guild_hint)
            if guild is None:
                await interaction.response.send_message(error or "I couldn't find that server.", ephemeral=True)
                return

        description = self.deps.describe_personalization(guild)
        await interaction.response.send_message(description, ephemeral=True)

    @personalize.command(name="set", description="Choose how Kaminiel personalizes announcements and heartbeats")
    @app_commands.describe(mode="How should Kaminiel address the server?")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Creator only", value="creator"),
            app_commands.Choice(name="Creator + server", value="both"),
            app_commands.Choice(name="Server only", value="server"),
        ]
    )
    @app_commands.guild_only()
    async def personalize_set(self, interaction: discord.Interaction, mode: app_commands.Choice[str]) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change personalization.",
                ephemeral=True,
            )
            return

        requested = mode.value.strip().lower()
        if requested not in {"creator", "both", "server"}:
            requested = self.deps.default_personalization

        previous = self.deps.personalization.get(guild.id, self.deps.default_personalization)

        if requested == self.deps.default_personalization:
            self.deps.personalization.pop(guild.id, None)
        else:
            self.deps.personalization[guild.id] = requested

        self.deps.save_personalization()

        if requested == previous:
            acknowledgement = {
                "creator": "Personalization was already creator-only—no changes needed!",
                "both": "Personalization already greeted the server while cuddling the creator.",
                "server": "Personalization was already focused on the whole server.",
            }[requested]
            message = acknowledgement
        else:
            confirmation = {
                "creator": "I'll keep heartbeats and announcements aimed right at Kamitchi now.",
                "both": "I'll greet the server and still speak softly to Kamitchi in every heartbeat.",
                "server": "I'll address the whole server and skip direct callouts to Kamitchi from now on.",
            }[requested]
            message = confirmation

        summary = self.deps.describe_personalization(guild)
        await interaction.response.send_message(f"{message}\n\n{summary}", ephemeral=True)

    @personalize.command(name="clear", description="Reset personalization to the default (creator-only)")
    @app_commands.describe(guild_hint="Server name or ID (only needed when using this command in DMs)")
    async def personalize_clear(self, interaction: discord.Interaction, guild_hint: Optional[str] = None) -> None:
        guild = interaction.guild
        if guild is None:
            guild, error = self.deps.resolve_guild_for_user(interaction.user, guild_hint)
            if guild is None:
                await interaction.response.send_message(error or "I couldn't find that server.", ephemeral=True)
                return
        elif not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change personalization.",
                ephemeral=True,
            )
            return

        if guild.id in self.deps.personalization:
            self.deps.personalization.pop(guild.id, None)
            self.deps.save_personalization()
            message = "Reset complete—I'll go back to cozy creator-only messaging."
        else:
            message = "Personalization was already at the default creator-only setting."

        await interaction.response.send_message(message, ephemeral=True)

    def _can_manage(self, interaction: discord.Interaction) -> bool:
        guild = interaction.guild
        if guild is None:
            return False
        user = interaction.user
        if self.deps.is_creator_user(user):
            return True
        return self.deps.can_manage_guild(guild, user)

    def _describe_destination(self, guild: discord.Guild, viewer: discord.abc.User) -> str:
        preference = self.deps.cache.get(guild.id)
        if not preference:
            return (
                f"Announcements for **{guild.name}** use automatic selection—I'll pick a channel I can speak in."
            )

        mode = preference.get("mode")
        if mode == "channel":
            channel_id = preference.get("channel_id")
            channel = guild.get_channel(channel_id) if isinstance(channel_id, int) else None
            if isinstance(channel, discord.TextChannel):
                return f"Announcements for **{guild.name}** go to {channel.mention}."
            return (
                f"I had a channel saved for **{guild.name}**, but I can't access it anymore. Please choose a new one."
            )

        if mode == "dm":
            user_id = preference.get("user_id")
            if isinstance(user_id, int):
                if user_id == viewer.id:
                    return f"Announcements for **{guild.name}** go straight to your DMs."
                member = guild.get_member(user_id) or self.bot.get_user(user_id)
                if member:
                    mention = getattr(member, "mention", self.deps.sanitize_display_name(getattr(member, "name", "friend")))
                    return f"Announcements for **{guild.name}** go to a DM with {mention}."
                return (
                    f"Announcements for **{guild.name}** are set to DM user ID {user_id}, but I can't reach them right now."
                )

        return f"Announcements for **{guild.name}** use automatic selection."


async def setup(bot: commands.Bot) -> None:  # pragma: no cover
    deps = getattr(bot, "_announce_deps", None)
    if deps is None:
        raise RuntimeError("Announcement dependencies were not attached to the bot before setup.")
    await bot.add_cog(AnnounceCog(bot, deps))
