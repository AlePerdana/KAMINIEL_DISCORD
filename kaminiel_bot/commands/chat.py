from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import ChatDependencies


class ChatCog(commands.Cog):
    """Slash command wrapper around Kaminiel's chat handler."""

    def __init__(self, bot: commands.Bot, deps: ChatDependencies) -> None:
        self.bot = bot
        self.deps = deps

    @app_commands.command(name="kaminiel", description="Chat with Kaminiel or ask for help")
    @app_commands.describe(message="What would you like to tell Kaminiel?")
    async def kaminiel(self, interaction: discord.Interaction, message: str) -> None:
        if not message.strip():
            await interaction.response.send_message(
                "Haii~ what do you want to ask me, nya~? 💞",
                ephemeral=True,
            )
            return

        await self.deps.handler(interaction, message.strip())


async def setup(bot: commands.Bot) -> None:  # pragma: no cover
    deps = getattr(bot, "_chat_deps", None)
    if deps is None:
        raise RuntimeError("Chat dependencies were not attached to the bot before setup.")
    await bot.add_cog(ChatCog(bot, deps))
