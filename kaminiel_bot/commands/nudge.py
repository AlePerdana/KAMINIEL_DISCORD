from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from .dependencies import NudgeDependencies

ALLOWED_NUDGE_DELIVERY_MODES: set[str] = {"dm", "server", "both"}


class NudgeCog(commands.Cog):
    """Slash commands for configuring Kaminiel's activity nudges."""

    nudge = app_commands.Group(name="nudge", description="Configure Kaminiel's activity nudges")

    def __init__(self, bot: commands.Bot, deps: NudgeDependencies) -> None:
        self.bot = bot
        self.deps = deps

    @nudge.command(name="show", description="Show how Kaminiel delivers activity nudges")
    @app_commands.guild_only()
    async def nudge_show(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None
        description = self.deps.describe_preference(guild)
        await interaction.response.send_message(description, ephemeral=True)

    @nudge.command(name="set_generic", description="Switch server nudges to gentle mode")
    @app_commands.guild_only()
    async def nudge_generic(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change activity nudges.",
                ephemeral=True,
            )
            return

        self.deps.server_preferences[guild.id] = {"mode": "generic"}
        self.deps.save_preferences()
        await interaction.response.send_message(
            "Okay! I'll keep nudges gentle—no one gets pinged directly in the server.",
            ephemeral=True,
        )

    @nudge.command(name="assign", description="Pick who Kaminiel should ping for nudges")
    @app_commands.describe(user="The user to ping when nudges fire")
    @app_commands.guild_only()
    async def nudge_assign(self, interaction: discord.Interaction, user: discord.Member) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change activity nudges.",
                ephemeral=True,
            )
            return

        clean_name = self.deps.sanitize_display_name(getattr(user, "display_name", getattr(user, "name", "friend")))
        self.deps.server_preferences[guild.id] = {
            "mode": "user",
            "user_id": user.id,
            "fallback_name": clean_name,
        }
        self.deps.save_preferences()
        await interaction.response.send_message(
            f"Got it! I'll ping {getattr(user, 'mention', clean_name)} whenever I send server nudges.",
            ephemeral=True,
        )

    @nudge.command(name="clear", description="Reset server nudges to default behavior")
    @app_commands.guild_only()
    async def nudge_clear(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change activity nudges.",
                ephemeral=True,
            )
            return

        self.deps.server_preferences.pop(guild.id, None)
        self.deps.save_preferences()
        await interaction.response.send_message(
            "Server nudges are back to their defaults (creators get pings, others gentle name mentions).",
            ephemeral=True,
        )

    @nudge.command(name="delivery", description="Choose how Kaminiel delivers activity nudges")
    @app_commands.describe(mode="Where should nudges be sent")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Direct Messages only", value="dm"),
            app_commands.Choice(name="Server channel only", value="server"),
            app_commands.Choice(name="Both DM and server", value="both"),
        ]
    )
    @app_commands.guild_only()
    async def nudge_delivery(self, interaction: discord.Interaction, mode: app_commands.Choice[str]) -> None:
        guild = interaction.guild
        assert guild is not None

        if not self._can_manage(interaction):
            await interaction.response.send_message(
                "You need Manage Server permissions (or be Kamitchi) to change where nudges are delivered.",
                ephemeral=True,
            )
            return

        requested = mode.value.strip().lower()
        if requested not in ALLOWED_NUDGE_DELIVERY_MODES:
            requested = "server"

        has_assignment = guild.id in self.deps.dm_assignments
        if requested in {"dm", "both"} and not has_assignment:
            await interaction.response.send_message(
                "I need someone to opt into DM nudges first. Use `/nudge dm_assign` so I know who to whisper before switching to DM delivery.",
                ephemeral=True,
            )
            return

        current = self.deps.delivery_preferences.get(guild.id, self.deps.default_delivery_mode)
        if current not in ALLOWED_NUDGE_DELIVERY_MODES:
            current = self.deps.default_delivery_mode

        if requested == self.deps.default_delivery_mode:
            self.deps.delivery_preferences.pop(guild.id, None)
        else:
            self.deps.delivery_preferences[guild.id] = requested

        self.deps.save_preferences()

        if requested == current:
            acknowledgement = {
                "dm": "Nudges were already set to arrive via DM only.",
                "server": "Nudges were already staying in the server channel.",
                "both": "Nudges were already going to both DM and the server.",
            }[requested]
            message = acknowledgement
        else:
            confirmation = {
                "dm": "I'll send nudges only through DMs now—no more public posts.",
                "server": "I'll keep nudges inside the server channel from now on.",
                "both": "I'll deliver nudges in both DMs and the server so no one misses them.",
            }[requested]
            message = confirmation

        summary = self.deps.describe_preference(guild)
        await interaction.response.send_message(f"{message}\n\n{summary}", ephemeral=True)

    @nudge.command(name="dm_assign", description="Opt someone into receiving DM nudges")
    @app_commands.describe(user="The person who should receive DM nudges")
    @app_commands.guild_only()
    async def nudge_dm_assign(self, interaction: discord.Interaction, user: discord.Member) -> None:
        guild = interaction.guild
        assert guild is not None

        if not (self._can_manage(interaction) or interaction.user.id == user.id):
            await interaction.response.send_message(
                "Only Kamitchi, server managers, or the person opting in can set DM nudges.",
                ephemeral=True,
            )
            return

        clean_name = self.deps.sanitize_display_name(getattr(user, "display_name", getattr(user, "name", "friend")))
        self.deps.dm_assignments[guild.id] = {
            "user_id": user.id,
            "display_name": clean_name,
        }
        self.deps.save_preferences()

        mention = getattr(user, "mention", clean_name)
        delivery_mode = self.deps.delivery_preferences.get(guild.id, self.deps.default_delivery_mode)
        if delivery_mode not in ALLOWED_NUDGE_DELIVERY_MODES:
            delivery_mode = self.deps.default_delivery_mode

        extra_hint = ""
        if delivery_mode == "server":
            extra_hint = " Use `/nudge delivery` if you'd like me to DM nudges now."

        await interaction.response.send_message(
            f"Okay! I'll DM nudges to {mention} whenever they fire.{extra_hint}",
            ephemeral=True,
        )

    @nudge.command(name="dm_clear", description="Stop sending DM nudges to the current recipient")
    @app_commands.guild_only()
    async def nudge_dm_clear(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        assert guild is not None

        assignment = self.deps.dm_assignments.get(guild.id)
        if not assignment:
            await interaction.response.send_message(
                "No one is currently getting DM nudges from me.",
                ephemeral=True,
            )
            return

        assigned_id = assignment.get("user_id")
        if not (self._can_manage(interaction) or interaction.user.id == assigned_id):
            await interaction.response.send_message(
                "Only Kamitchi, server managers, or the current DM recipient can clear the assignment.",
                ephemeral=True,
            )
            return

        self.deps.dm_assignments.pop(guild.id, None)

        delivery_mode = self.deps.delivery_preferences.get(guild.id, self.deps.default_delivery_mode)
        if delivery_mode not in ALLOWED_NUDGE_DELIVERY_MODES:
            delivery_mode = self.deps.default_delivery_mode

        if delivery_mode in {"dm", "both"}:
            if self.deps.default_delivery_mode == "server":
                self.deps.delivery_preferences.pop(guild.id, None)
            else:
                self.deps.delivery_preferences[guild.id] = "server"

        self.deps.save_preferences()

        await interaction.response.send_message(
            "Got it! I won't DM nudges anymore, and I'll stick to the server channel until someone opts back in.",
            ephemeral=True,
        )

    def _can_manage(self, interaction: discord.Interaction) -> bool:
        guild = interaction.guild
        if guild is None:
            return False
        user = interaction.user
        if self.deps.is_creator_user(user):
            return True
        return self.deps.can_manage_guild(guild, user)


async def setup(bot: commands.Bot) -> None:  # pragma: no cover
    deps = getattr(bot, "_nudge_deps", None)
    if deps is None:
        raise RuntimeError("Nudge dependencies were not attached to the bot before setup.")
    await bot.add_cog(NudgeCog(bot, deps))
