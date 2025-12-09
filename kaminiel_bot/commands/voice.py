from __future__ import annotations

from typing import Optional, Sequence, Union

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import VoiceDependencies

VoiceChannelLike = Union[discord.VoiceChannel, discord.StageChannel]
VOICE_CHANNEL_TYPES = (discord.VoiceChannel, discord.StageChannel)


class VoiceCog(commands.Cog):
    """Slash commands for managing Kaminiel's voice playback destination."""

    voice = app_commands.Group(name="voice", description="Configure Kaminiel's voice playback")

    def __init__(self, bot: commands.Bot, deps: VoiceDependencies) -> None:
        self.bot = bot
        self.deps = deps
        self.voice_flags = deps.voice_flags
        self.creator_ids: Sequence[int] = deps.creator_user_ids
        self.disabled_guilds = deps.disabled_guilds
        self.save_disabled = deps.save_disabled

    @voice.command(name="show", description="Show where Kaminiel will speak")
    @app_commands.guild_only()
    async def voice_show(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        channel_id = self.deps.channel_cache.get(guild.id)
        disabled_note = ""
        if guild.id in self.disabled_guilds:
            disabled_note = "\n(Voice playback is currently disabled; use `/voice enable` when you're ready for me to speak again.)"
        if channel_id:
            channel = guild.get_channel(channel_id)
            if isinstance(channel, VOICE_CHANNEL_TYPES):
                await interaction.response.send_message(
                    f"I'll speak in {channel.mention} when I go voice.{disabled_note}",
                    ephemeral=True,
                )
                return
            self.deps.channel_cache.pop(guild.id, None)
            self.deps.save_cache()

        await interaction.response.send_message(
            "No voice channel is saved. I'll look for Kamitchi's current voice channel instead."
            f"{disabled_note}",
            ephemeral=True,
        )

    @voice.command(name="set", description="Choose which voice channel Kaminiel should use")
    @app_commands.describe(channel="Voice or stage channel to use for playback")
    @app_commands.guild_only()
    async def voice_set(self, interaction: discord.Interaction, channel: VoiceChannelLike) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You'll need the Manage Server permission (or be Kamitchi) to change my voice channel.",
                ephemeral=True,
            )
            return

        self.deps.channel_cache[guild.id] = channel.id
        self.deps.save_cache()
        await interaction.response.send_message(
            f"Noted! I'll hop into {channel.mention} whenever I speak out loud.",
            ephemeral=True,
        )

    @voice.command(name="clear", description="Forget the saved voice channel")
    @app_commands.guild_only()
    async def voice_clear(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You'll need the Manage Server permission (or be Kamitchi) to change my voice channel.",
                ephemeral=True,
            )
            return

        if guild.id in self.deps.channel_cache:
            self.deps.channel_cache.pop(guild.id, None)
            self.deps.save_cache()
            message = "Cleared the saved voice channel. I'll try to follow Kamitchi's voice channel if I see him online."
        else:
            message = "I don't have a saved voice channel yet."
        await interaction.response.send_message(message, ephemeral=True)

    @voice.command(name="join", description="Invite Kaminiel into the configured voice channel")
    @app_commands.guild_only()
    async def voice_join(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if guild.id in self.disabled_guilds:
            await interaction.response.send_message(
                "Voice playback is disabled for this server. Use `/voice enable` when you're ready for me to join again.",
                ephemeral=True,
            )
            return

        if not self._dependencies_ready():
            await interaction.response.send_message(
                self.deps.dependency_warning_message,
                ephemeral=True,
            )
            return

        channel = self._resolve_voice_channel(interaction)
        if channel is None:
            await interaction.response.send_message(
                "I don't know which voice channel to join yet. Try `/voice set` or hop into a voice channel and ask me again.",
                ephemeral=True,
            )
            return

        lock = self.deps.get_voice_lock(guild.id)
        async with lock:
            self.deps.cancel_voice_disconnect(guild.id)
            voice_client = guild.voice_client
            try:
                if voice_client and voice_client.is_connected():
                    if voice_client.channel == channel:
                        await interaction.response.send_message(
                            f"I'm already waiting in {channel.mention}.",
                            ephemeral=True,
                        )
                        return
                    await voice_client.move_to(channel)  # type: ignore[arg-type]
                else:
                    voice_client = await channel.connect(reconnect=True)  # type: ignore[arg-type]
            except RuntimeError as exc:
                if "PyNaCl" in str(exc):
                    self.voice_flags["dependencies_ready"] = False
                    self.voice_flags["warning_emitted"] = False
                    await interaction.response.send_message(
                        self.deps.dependency_warning_message,
                        ephemeral=True,
                    )
                else:
                    await interaction.response.send_message(
                        "Something blocked me from joining voice just now. Could you double-check my permissions?",
                        ephemeral=True,
                    )
                return
            except (discord.ClientException, discord.Forbidden, discord.HTTPException):
                await interaction.response.send_message(
                    "I couldn't slip into that voice channel—maybe I lack permission?",
                    ephemeral=True,
                )
                return

        await interaction.response.send_message(
            f"All tucked into {channel.mention}. I'll stay on standby there now!",
            ephemeral=True,
        )

    @voice.command(name="disconnect", description="Ask Kaminiel to leave voice")
    @app_commands.guild_only()
    async def voice_disconnect(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        voice_client = guild.voice_client
        if not voice_client or not voice_client.is_connected():
            await interaction.response.send_message(
                "I'm not in a voice channel at the moment.",
                ephemeral=True,
            )
            return

        lock = self.deps.get_voice_lock(guild.id)
        async with lock:
            self.deps.cancel_voice_disconnect(guild.id)
            try:
                await voice_client.disconnect(force=True)
            except (discord.ClientException, discord.HTTPException):
                await interaction.response.send_message(
                    "I tried to slip out, but something held me in place. Maybe check my permissions?",
                    ephemeral=True,
                )
                return

        await interaction.response.send_message(
            "Left the voice channel. Call me back anytime with `/voice join`.",
            ephemeral=True,
        )

    @voice.command(name="disable", description="Keep Kaminiel out of voice channels until re-enabled")
    @app_commands.guild_only()
    async def voice_disable(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You'll need the Manage Server permission (or be Kamitchi) to change my voice settings.",
                ephemeral=True,
            )
            return

        if guild.id in self.disabled_guilds:
            await interaction.response.send_message(
                "I'm already staying out of voice here. Use `/voice enable` when you want me back in.",
                ephemeral=True,
            )
            return

        self.disabled_guilds.add(guild.id)
        self.save_disabled()
        await self._force_disconnect(guild)
        await interaction.response.send_message(
            "Okay, I'll stay out of voice channels until you run `/voice enable`.",
            ephemeral=True,
        )

    @voice.command(name="enable", description="Allow Kaminiel to join voice again")
    @app_commands.guild_only()
    async def voice_enable(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You'll need the Manage Server permission (or be Kamitchi) to change my voice settings.",
                ephemeral=True,
            )
            return

        if guild.id not in self.disabled_guilds:
            await interaction.response.send_message(
                "I'm already allowed to join voice. Call me anytime with `/voice join`.",
                ephemeral=True,
            )
            return

        self.disabled_guilds.discard(guild.id)
        self.save_disabled()
        await interaction.response.send_message(
            "Voice playback re-enabled. Use `/voice join` when you'd like me in channel again!",
            ephemeral=True,
        )

    def _can_manage(self, interaction: discord.Interaction) -> bool:
        guild = interaction.guild
        if guild is None:
            return False
        member = guild.get_member(interaction.user.id)
        if member is None:
            return False
        perms = member.guild_permissions
        if perms.manage_guild or perms.administrator:
            return True
        return interaction.user.id in self.creator_ids

    def _dependencies_ready(self) -> bool:
        ready = self.voice_flags.get("dependencies_ready", True)
        if not ready and not self.voice_flags.get("warning_emitted", False):
            print(self.deps.dependency_warning_message)
            self.voice_flags["warning_emitted"] = True
        return bool(ready)

    def _resolve_voice_channel(self, interaction: discord.Interaction) -> Optional[VoiceChannelLike]:
        guild = interaction.guild
        assert guild is not None

        saved_channel_id = self.deps.channel_cache.get(guild.id)
        if saved_channel_id:
            saved_channel = guild.get_channel(saved_channel_id)
            if isinstance(saved_channel, VOICE_CHANNEL_TYPES):
                return saved_channel
            self.deps.channel_cache.pop(guild.id, None)
            self.deps.save_cache()

        if isinstance(interaction.user, discord.Member):
            author_voice = getattr(interaction.user, "voice", None)
            if author_voice and isinstance(author_voice.channel, VOICE_CHANNEL_TYPES):
                channel = author_voice.channel
                if interaction.user.id not in self.creator_ids:
                    self.deps.channel_cache[guild.id] = channel.id
                    self.deps.save_cache()
                return channel

        for creator_id in self.creator_ids:
            member = guild.get_member(creator_id)
            if member and member.voice and isinstance(member.voice.channel, VOICE_CHANNEL_TYPES):
                return member.voice.channel

        return None

    async def _force_disconnect(self, guild: discord.Guild) -> bool:
        voice_client = guild.voice_client
        if not voice_client or not voice_client.is_connected():
            return False

        lock = self.deps.get_voice_lock(guild.id)
        async with lock:
            self.deps.cancel_voice_disconnect(guild.id)
            try:
                await voice_client.disconnect(force=True)
            except (discord.ClientException, discord.HTTPException):
                return False
        return True


async def setup(bot: commands.Bot) -> None:  # pragma: no cover
    deps = getattr(bot, "_voice_deps", None)
    if deps is None:
        raise RuntimeError("Voice dependencies were not attached to the bot before setup.")
    await bot.add_cog(VoiceCog(bot, deps))
