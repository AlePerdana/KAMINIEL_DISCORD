# Kaminiel Discord Bot

A cutesy D- `!kaminiel## Crea## Features

- `!kaminiel <message>` — Chat with Kaminiel using Gemini.
- Warm wake-up announcement when she logs in, including the last time she spoke and the current Jakarta (WIB) time.
- Gentle heartbeat check-ins every 15 minutes (configurable) so the chat never feels lonely.
- Wake-up and goodnight rituals rotate through cozy prompt variations, including the current Jakarta (WIB) time so everyone knows when she arrived or left.

If you want Kaminiel to recognize Kamitchi/Ale as her beloved creator (even when he isn't the server owner), set one or both of the following in `.env`:

- `KAMINIEL_CREATOR_IDS` – Comma- or semicolon-separated list of numeric Discord user IDs.
- `KAMINIEL_CREATOR_NAMES` – Optional comma-separated list of display/nick names to match (case-insensitive).

When these are present she'll switch into an extra-cutesy tone for Kamitchi, while staying warm and natural for everyone else and still treating the actual guild owner respectfully.

### Creator protection

Kaminiel actively protects her creator with two safety layers:

1. **Command refusal**: If anyone (including the creator) tries to use `mute`, `deafen`, or `timeout` commands on the creator, Kaminiel will refuse with a loving message.
2. **Auto-removal**: If the creator somehow gets muted, deafened, or timed out (via Discord's UI or another bot), Kaminiel will instantly remove it and post a shield notification in the server's system channel.
3. **Tantrum shield**: Manual tantrums can never target Kamitchi, and any "clanker" or slander aimed at him (or at Kaminiel) triggers an automatic five-minute stomp-fest—unless it came from Kamitchi himself, in which case she just wilts and scolds him lovingly.

This ensures the creator always has full voice and can never be silenced.e>` — Chat with Kaminiel using Gemini.
- Natural-language pings (e.g., "Kaminiel please ping @User" or just "Kaminiel ping @User").
- Gentle moderation helpers for **mute/unmute**, **deafen/undeafen**, **timeout**, and **timeout removal** (e.g., "Kaminiel unmute @User" or "remove timeout @User").
- Cozy art mode – ask with `!kaminiel draw <prompt>` (or `image`, `paint`, etc.) and she'll send back an attachment.
- Context-aware emoji reactions whenever you use `!kaminiel` or mention Kaminiel directly.
- Passive listening for requests addressed to "Kaminiel" without needing the `!kaminiel` prefix. companion that proxies messages to the Gemini API, reacts with heart emojis, and helps with light moderation tasks.

## Requirements

- Python 3.10 or newer (tested with 3.13)
- A Discord bot token with the following **privileged gateway intents enabled** in the [Discord Developer Portal](https://discord.com/developers/applications/):
  - **Message Content Intent** (required so Kaminiel can read chat messages)
  - **Server Members Intent** (required for mute, deafen, and timeout helpers)
- A Google Gemini API key
- `ffmpeg` installed and available on your `PATH` (or set `KAMINIEL_FFMPEG_PATH` to its location) so voice playback can run.
- [gTTS](https://pypi.org/project/gTTS/) (installed via `requirements.txt`) for lightweight speech synthesis.
- [PyNaCl](https://pypi.org/project/PyNaCl/) (installed via `requirements.txt`) so Discord voice connections can initialize.

If you see `discord.errors.PrivilegedIntentsRequired` when starting the bot, double-check that both intents above are enabled and saved in the portal before you try again.

## Environment Setup

1. Copy `.env.example` to `.env` and fill in your secrets.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the bot:

```powershell
venv\Scripts\activate
python bot.py
```

Kaminiel should greet you in the console once she's connected.

## Features

- `!kaminiel <message>` — Chat with Kaminiel using Gemini.
- Natural-language pings (e.g., “Kaminiel please ping @User” or “Kaminiel ping Bob”).
- Gentle moderation helpers for **mute/unmute**, **deafen/undeafen**, **timeout**, and **timeout removal**.
- Flexible member references – mention someone or just type their username/display name (no @ autocomplete needed).
- Cozy art mode – ask with `!kaminiel draw <prompt>` (or `image`, `paint`, etc.) and she'll send back an attachment.
- Context-aware emoji reactions whenever you use `!kaminiel` or mention Kaminiel directly.
- Passive listening for requests addressed to “Kaminiel” without needing the `!kaminiel` prefix.
- Warm wake-up announcements when she logs in, including the last time she spoke and the current Jakarta (WIB) time.
- Lightweight voice-overs: she can hop into a configured voice channel and read every message aloud with a soft female TTS voice (`!kaminiel_voice`).
- Gentle Gemini-crafted heartbeat check-ins every few minutes (default 15) with a worry meter that escalates after 60 minutes of silence, then again at 90, 105, 110, and 111 minutes—with the last two tiers pinging Kamitchi directly.
- Time-aware persona: every reply knows the current day, date, and WIB timestamp.
- `!kaminiel_help` — see a quick list of the commands she supports.
- `!kaminiel_nudge server ...` — control whether server activity nudges stay generic or ping a specific caretaker.
- `!kaminiel_heartbeat <mode>` — switch heartbeat delivery between the guild destination (`generic`), DMing Kamitchi (`creator`), or doing both.
- Activity-aware break nudges that pull fresh wording from Gemini so reminders feel lively instead of repetitive.
- Manual tantrums on command — try `!kaminiel throw a tantrum on @Owner because cuddle time was skipped for 30 minutes` to have her stomp and complain on a timer.
- `!kaminiel stop throwing tantrum` — the original requester can cancel their own tantrum, while Kamitchi can hush **all** tantrums instantly with a single command.
- Automatic tantrum trigger if anyone hurls "clanker" or other slander at Kamitchi or Kaminiel; apologizing sincerely will make her stop stomping.
- `!kaminiel stop my reminders` — cancel your latest scheduled reminder (add “all” to clear every pending one).
- `!kaminiel show my current reminders` — list the reminders she’s still holding for you with their remaining timers.

## Voice playback

- Use `!kaminiel_voice set <voice channel>` to tell Kaminiel which voice room to join when she speaks.
- `!kaminiel_voice show` reveals the saved channel, and `!kaminiel_voice clear` removes it—after that she shadow-steps into Kamitchi’s current voice room and keeps tailing him whenever he moves.
- `!kaminiel_voice join` plants her in the chosen voice channel on standby (if you’ve cleared the saved channel she’ll stay latched to Kamitchi after that), and `!kaminiel_voice disconnect` lets her slip back out when you're done.
- Every text reply she sends is mirrored through gTTS, giving you a gentle, feminine voice line in the chosen voice channel.
- If no specific channel is configured, she tails Kamitchi across voice rooms and stays beside him instead of disconnecting. Once she can’t find him anywhere for a full minute, she slips out of voice and posts a soft “my creator left, I’ll go too” note.
- Ensure `ffmpeg` is installed and reachable; you can point to a custom binary with the `KAMINIEL_FFMPEG_PATH` environment variable if needed.
- Install PyNaCl (included in `requirements.txt`) if the bot reports that voice support is unavailable.

## Creator recognition

If you want Kaminiel to recognize Kamitchi/Ale as her beloved creator (even when he isn’t the server owner), set one or both of the following in `.env`:

- `KAMINIEL_CREATOR_IDS` – Comma- or semicolon-separated list of numeric Discord user IDs.
- `KAMINIEL_CREATOR_NAMES` – Optional comma-separated list of display/nick names to match (case-insensitive).

When these are present she’ll switch into an extra-cutesy tone for Kamitchi, while staying warm and natural for everyone else and still treating the actual guild owner respectfully.

## Image generation

- Use `!kaminiel draw <prompt>` (or `image`, `paint`, `illustrate`, etc.) to ask for artwork.
- The same keywords work in passive wake-ups (e.g., “Kaminiel, draw me a pastel sunset”).
- Kaminiel uploads the artwork as a PNG attachment.
- Customize the model with `GEMINI_IMAGE_MODEL` (defaults to `imagen-3.0-generate`).

## Time awareness & heartbeats

- Kaminiel greets every guild she belongs to when she logs in, sharing the current Jakarta/WIB time and the last moment she spoke there.
- Every `KAMINIEL_HEARTBEAT_MINUTES` (default **15**) she posts a cozy check-in message to keep conversations warm. Each heartbeat now asks Gemini for fresh wording and automatically falls back to her handcrafted templates if the API is unavailable. Set the variable to `0` to disable the heartbeat loop entirely.
- Customize how those heartbeats arrive with `!kaminiel_heartbeat`: check the current mode with `show`, switch to server-only `generic`, DM-only `creator`, or doubled-up `both`.
- Wake-up and bedtime announcements are the only moments that mention the exact WIB time, so the timeline stays special instead of spammy.
- If Kamitchi stays quiet, her worry meter now ramps at **60 minutes**, then at **90**, **105**, **110**, and **111** minutes of silence. The 5-minute tier starts pinging him directly, and the 1-minute tier is an all-out desperate plea—once he answers, she snaps back to calm immediately.

## Graceful shutdown

- Pressing <kbd>Ctrl</kbd>+<kbd>C</kbd> (or sending a keyboard interrupt) makes Kaminiel say goodbye to every guild before disconnecting.
- Prefer a soft landing by typing `sleep` (or `goodnight`, `quit`, `exit`, etc.) into the console prompt—she'll tuck herself in and log off for you.
- Farewell notes also include the current WIB time so everyone knows when she signed off.

## Troubleshooting

- **Privileged intents error on startup**: Make sure both Message Content and Server Members intents are toggled on in the portal and click *Save Changes*.
- **Permission errors in Discord**: Invite the bot with permissions to read message history, mute members, deafen members, and moderate members (timeout).
- **Gemini model errors**: Update `GEMINI_MODEL` in `.env` to a currently supported model name.
