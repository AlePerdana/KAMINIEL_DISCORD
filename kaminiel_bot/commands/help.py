from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import HelpDependencies


class HelpCog(commands.Cog):
    """Slash command for displaying Kaminiel's command summary."""

    def __init__(self, bot: commands.Bot, deps: HelpDependencies) -> None:
        self.bot = bot
        self.deps = deps

    @app_commands.command(name="kaminiel_help", description="Show Kaminiel's command list")
    async def kaminiel_help(self, interaction: discord.Interaction) -> None:
        lines = list(self.deps.generate_help_lines())
        if not lines:
            lines = ["I don't have any commands to show right now."]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)


async def setup(bot: commands.Bot) -> None:  # pragma: no cover
    deps = getattr(bot, "_help_deps", None)
    if deps is None:
        raise RuntimeError("Help dependencies were not attached to the bot before setup.")
    await bot.add_cog(HelpCog(bot, deps))
