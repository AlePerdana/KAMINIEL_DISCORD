from __future__ import annotations

from typing import TYPE_CHECKING

from .dependencies import (
    AnnounceDependencies,
    ChatDependencies,
    CommandDependencies,
    HeartbeatDependencies,
    HelpDependencies,
    NudgeDependencies,
    VoiceDependencies,
)

if TYPE_CHECKING:
    from discord.ext import commands

from .heartbeat import HeartbeatCog
from .nudge import NudgeCog
from .announce import AnnounceCog
from .voice import VoiceCog
from .chat import ChatCog
from .help import HelpCog

__all__ = [
    "CommandDependencies",
    "HeartbeatDependencies",
    "NudgeDependencies",
    "AnnounceDependencies",
    "VoiceDependencies",
    "ChatDependencies",
    "HelpDependencies",
    "setup",
]


async def setup(bot: "commands.Bot", deps: CommandDependencies) -> None:
    """Register all Kaminiel command cogs with the provided dependencies."""

    await bot.add_cog(HeartbeatCog(bot, deps.heartbeat))
    await bot.add_cog(NudgeCog(bot, deps.nudge))
    await bot.add_cog(AnnounceCog(bot, deps.announce))
    await bot.add_cog(VoiceCog(bot, deps.voice))
    await bot.add_cog(ChatCog(bot, deps.chat))
    await bot.add_cog(HelpCog(bot, deps.help))
