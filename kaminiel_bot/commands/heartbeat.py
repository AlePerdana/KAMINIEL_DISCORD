from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import HeartbeatDependencies


class HeartbeatCog(commands.Cog):
    """Slash commands for configuring heartbeat delivery."""

    heartbeat = app_commands.Group(name="heartbeat", description="Adjust Kaminiel's heartbeat delivery mode")

    def __init__(self, bot: commands.Bot, deps: HeartbeatDependencies) -> None:
        self.bot = bot
        self.deps = deps

    @heartbeat.command(name="show", description="Show the current heartbeat delivery mode")
    @app_commands.guild_only()
    async def heartbeat_show(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None  # enforced by guild_only
        description = self.deps.describe_preference(guild)
        await interaction.response.send_message(description, ephemeral=True)

    @heartbeat.command(name="set", description="Set the heartbeat delivery mode")
    @app_commands.describe(mode="Choose how heartbeats are delivered")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Creator DM", value="creator"),
            app_commands.Choice(name="Server Channel", value="server"),
            app_commands.Choice(name="Both (DM + Server)", value="both"),
        ]
    )
    @app_commands.guild_only()
    async def heartbeat_set(
        self,
        interaction: discord.Interaction,
        mode: app_commands.Choice[str],
    ) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage_heartbeats(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to adjust heartbeats.",
                ephemeral=True,
            )
            return

        destination = mode.value.strip().lower()
        if destination not in {"creator", "server", "both"}:
            destination = "creator"

        previous = self.deps.preferences.get(guild.id)
        effective_previous = previous if isinstance(previous, str) else None
        if effective_previous not in {"creator", "server", "both"}:
            effective_previous = "creator"
        already_set = effective_previous == destination

        if destination == "creator":
            self.deps.preferences.pop(guild.id, None)
        else:
            self.deps.preferences[guild.id] = destination
        self.deps.save_preferences()

        templates = {
            "creator": (
                "Heartbeats are already cuddled up in Kamitchi's DMs—nothing to change!",
                "I'll keep every heartbeat between Kamitchi and me, delivered straight to his DMs.",
            ),
            "server": (
                "Heartbeats are already shimmering in the server for Kamitchi—no tweaks needed!",
                "I'll share each heartbeat in the server so everyone can see how tenderly I watch over Kamitchi.",
            ),
            "both": (
                "Heartbeats are already traveling through both Kamitchi's DMs and the server—double the affection already in place!",
                "I'll send every heartbeat twice: once privately to Kamitchi and once in the server so the whole family can cheer him on.",
            ),
        }
        already_text, update_text = templates[destination]
        message = already_text if already_set else update_text

        await interaction.response.send_message(message, ephemeral=True)

    @heartbeat.command(name="reset", description="Reset heartbeat delivery to the default guild channel")
    @app_commands.guild_only()
    async def heartbeat_reset(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage_heartbeats(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to adjust heartbeats.",
                ephemeral=True,
            )
            return

        if guild.id in self.deps.preferences:
            self.deps.preferences.pop(guild.id, None)
            self.deps.save_preferences()

        await interaction.response.send_message(
            "Reset complete—I'll slip back to the default of whispering heartbeats directly to Kamitchi's DMs.",
            ephemeral=True,
        )

    def _can_manage_heartbeats(self, interaction: discord.Interaction) -> bool:
        guild = interaction.guild
        if guild is None:
            return False
        user = interaction.user
        if self.deps.is_creator_user(user):
            return True
        return self.deps.can_manage_guild(guild, user)


async def setup(bot: commands.Bot) -> None:  # pragma: no cover - for older extension loading style
    deps = getattr(bot, "_heartbeat_deps", None)
    if deps is None:
        raise RuntimeError("Heartbeat dependencies were not attached to the bot before setup.")
    await bot.add_cog(HeartbeatCog(bot, deps))
