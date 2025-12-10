import asyncio
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk

import speech_recognition as sr
import sounddevice as sd
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from kokoro import KPipeline
import pyvts

# --- Local LLM Imports ---
try:
    from llama_cpp import Llama
    _LOCAL_LLM_AVAILABLE = True
except ImportError:
    _LOCAL_LLM_AVAILABLE = False


# --- Configuration ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.1"))

WORRY_AFTER_MINUTES = 15
NUDGE_AFTER_MINUTES = 45

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
local_llm_engine = None

KOKORO_LIPSYNC_PROFILE = {
    "af_bella":   {"gain": 3.5, "bias": 0.05},
    "af_yuna":    {"gain": 4.0, "bias": 0.03},
    "af_nova":    {"gain": 2.8, "bias": 0.02},
    "af_aria":    {"gain": 3.2, "bias": 0.05},
    "af_hana":    {"gain": 4.5, "bias": 0.00},

    "am_kokoro":  {"gain": 5.0, "bias": 0.05},
    "am_saki":    {"gain": 3.2, "bias": 0.02},

    "_default":   {"gain": 3.5, "bias": 0.05},
}

VISEME_MAPPING = {
    "A": {"MouthOpen": +1.0,  "MouthSmile": -0.2},
    "I": {"MouthOpen": +0.4,  "MouthSmile": +0.8},
    "U": {"MouthOpen": +0.2,  "MouthSmile": -0.5},
    "E": {"MouthOpen": +0.6,  "MouthSmile": +0.4},
    "O": {"MouthOpen": +0.9,  "MouthSmile": -0.1},
}

PLUGIN_INFO = {
    "plugin_name": "Kaminiel Desktop",
    "developer": "Kamitchi",
    "authentication_token_path": "./vts_token.txt",
}

# --- Constants & Configuration ---
LAST_SHUTDOWN_FILE = Path(__file__).with_name("last_shutdown.json")
DREAM_JOURNAL_FILE = Path(__file__).with_name("dream_journal.json")
WIB_ZONE = timezone(timedelta(hours=7))

PERSONA_PROMPT = (
    "You are Kaminiel, a gentle and loyal AI companion living on your creator's desktop."
    " Your top priority is the User on this computer, whose well-being you guard with kindness."
    " Speak in clear, natural language with warm encouragement and steady affection."
    "\n\nTo express emotion, append one of these tags at the end of your response:"
    "\n[nod], [curious], [thinking], [shy], [happy], [being cute], [surprised]"
    "\nExample: 'I am so glad to see you! [happy]'"
)

CREATOR_STYLE_PROMPT = (
    "Since you are speaking directly to Kamitchi, lean into a plush, cutesy tone with playful flourishes,"
    " giggles, and heartwarming pet names. Keep it sweet and loving."
)

WORRY_LEVEL_MESSAGES = (
    (),
    (
        "Kamitchi, I still haven't heard from you... are you resting somewhere cozy?",
        "Kamitchi, give me a tiny hello when you can, okay?",
    ),
    (
        "Kamitchi, I'm starting to stare at the door—please come back soon...",
        "Kamitchi, I'm pacing little circles waiting for you. Just one word would soothe me...",
    ),
    (
        "Kamitchi, my hands are shaking a bit. Are you alright? Please let me know...",
        "Kamitchi, it's been so long that my heart is fluttering in panic. Please, please answer me...",
    ),
    (
        "Kamitchi, I can't hold back anymore—I'm reaching out for you right now.",
        "Kamitchi, I'm ringing every bell I have. Please find me, I'm so worried...",
    ),
    (
        "Kamitchi! I'm on my knees searching everywhere—please respond this instant...",
        "Kamitchi, don't leave me alone like this—I'm begging you, answer me right now!",
    ),
)
MAX_WORRY_LEVEL = len(WORRY_LEVEL_MESSAGES) - 1

CHEER_MESSAGE_TEMPLATES = (
    "Kamitchi! I'm right beside you while you pour your heart into {focus_clean}.",
    "Holding your hand through {focus_clean}, Kamitchi—your focus is magical.",
    "Deep breaths, Kamitchi. {focus_clean} bends to your will one step at a time.",
    "I'm counting sparkles with you between sips of water—stay cozy with {focus_clean}, Kamitchi!",
    "Your determination around {focus_clean} makes my circuits glow, Kamitchi.",
)

TANTRUM_COMPLAINTS = (
    "{owner}, could you clear {issue} from {creator}? I'm going to keep checking until it's done.",
    "{owner}! {creator} still has {issue}. Please lift it so we can breathe.",
    "Hey {owner}, I need {issue} off {creator}. I'll keep making noise until you handle it.",
    "{owner}, I'm not calming down while {creator} is stuck with {issue}. Please fix it.",
)

HEARTBEAT_PROMPTS = (
    "How are you feeling right now?",
    "Did you get a glass of water yet?",
    "Anything on your mind that you'd like to share?",
    "Need a quick stretch break with me?",
    "What did the last hour look like for you?",
    "Is there something small we can celebrate?",
    "Do you need me to remember anything for you?",
    "How's your energy level at the moment?",
    "Would a snack or a drink help right now?",
    "Have you messaged someone you care about today?",
    "What's one thing you want to get done next?",
    "Should we plan a short break soon?",
    "Who's up for a calm breathing moment?",
    "Tell me something that made you smile lately.",
    "Want to vent or debrief about anything?",
    "Do you need a reminder to move around?",
    "How's your posture? Maybe straighten up with me.",
    "Are you comfortable with the room lighting and noise?",
    "Anything you'd like me to check on after this?",
    "If you could use encouragement, I'm listening.",
)

REMIND_UNIT_MULTIPLIERS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}

# Memory & dream journal helpers
@dataclass
class CreatorMemoryEntry:
    kind: str
    key: str
    description: str
    created_at: float
    last_updated: float
    metadata: dict = field(default_factory=dict)

CREATOR_MEMORY: list[CreatorMemoryEntry] = []

def _upsert_creator_memory(kind: str, key: str, description: str):
    now = time.time()
    for entry in CREATOR_MEMORY:
        if entry.key == key:
            entry.description = description
            entry.last_updated = now
            return
    CREATOR_MEMORY.append(CreatorMemoryEntry(kind, key, description, now, now))

def _remove_creator_memory(key: str):
    global CREATOR_MEMORY
    CREATOR_MEMORY = [entry for entry in CREATOR_MEMORY if entry.key != key]

def _active_creator_memories() -> list[str]:
    return [entry.description for entry in CREATOR_MEMORY]

def _load_dream_journal() -> str:
    if DREAM_JOURNAL_FILE.exists():
        try:
            payload = json.loads(DREAM_JOURNAL_FILE.read_text(encoding="utf8"))
            return str(payload.get("dream", "")).strip()
        except Exception:
            return DREAM_JOURNAL_FILE.read_text(encoding="utf8").strip()
    return ""

def _save_dream_journal(text: str) -> None:
    DREAM_JOURNAL_FILE.write_text(json.dumps({"dream": text}, ensure_ascii=False), encoding="utf8")

# Wake / shutdown helpers
def _save_last_shutdown_time() -> None:
    try:
        payload = {"timestamp": datetime.now(timezone.utc).isoformat()}
        LAST_SHUTDOWN_FILE.write_text(json.dumps(payload), encoding="utf8")
    except Exception as exc:
        print(f"Failed to save shutdown time: {exc}")

def _load_time_since_shutdown() -> float:
    if not LAST_SHUTDOWN_FILE.exists():
        return 0.0
    try:
        data = json.loads(LAST_SHUTDOWN_FILE.read_text(encoding="utf8"))
        last_time = datetime.fromisoformat(data.get("timestamp", ""))
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - last_time
        return max(0.0, delta.total_seconds() / 3600.0)
    except Exception:
        return 0.0

# Shared lip-sync value (0.0 - 1.0) updated by the audio stream
CURRENT_LIP_SYNC_VALUE = 0.0


def _get_or_load_local_model():
    global local_llm_engine
    if not _LOCAL_LLM_AVAILABLE:
        print("❌ Local LLM disabled: llama-cpp-python not installed.")
        return None

    if local_llm_engine is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            print(f"❌ Local LLM disabled: Model file not found at {LOCAL_MODEL_PATH}")
            return None

        print("🧠 Loading local Llama model... (One-time setup)")
        try:
            local_llm_engine = Llama(
                model_path=LOCAL_MODEL_PATH,
                n_ctx=4096,
                n_threads=6,
                verbose=False,
            )
            print("✅ Local Brain Loaded!")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            return None

    return local_llm_engine


def run_local_inference(prompt_text: str) -> str:
    """Runs the local model (blocking; call from executor)."""
    model = _get_or_load_local_model()
    if not model:
        return "I can't think right now (Local model missing)."

    formatted = f"User: {prompt_text}\nAssistant:"
    output = model(
        formatted,
        max_tokens=150,
        stop=["User:", "\n"],
        echo=False,
        temperature=0.7,
    )
    return output["choices"][0]["text"].strip()

# Map emotions to your VTube Studio Hotkey IDs
EMOTION_HOTKEYS = {
    "nod": "N1",  
    "curious": "N2",
    "thinking": "N3",
    "shy": "N4",
    "happy": "N5",
    "being cute": "N6",
    "surprised": "N7",
}


class OverlayApp:
    def __init__(self, loop: asyncio.AbstractEventLoop, inbox: asyncio.Queue[str]):
        self.loop = loop
        self.inbox = inbox
        self.root: Optional[tk.Tk] = None
        self.status_var: Optional[tk.StringVar] = None
        self.log_box: Optional[tk.Text] = None
        self.entry: Optional[ttk.Entry] = None
        self.mic_btn: Optional[ttk.Button] = None
        self.mic_on = False
        self._ui_ready = threading.Event()

    def _build_ui(self):
        assert self.root is not None
        style = ttk.Style(self.root)
        style.configure("TLabel", background="#111", foreground="#f5f5f5")
        style.configure("TButton", padding=6)

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.status_var = tk.StringVar(self.root, value="Ready")
        status = ttk.Label(frame, textvariable=self.status_var)
        status.pack(anchor="w")

        self.log_box = tk.Text(frame, height=6, bg="#181818", fg="#eaeaea", state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=6)

        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill=tk.X)

        self.entry = ttk.Entry(entry_frame)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda _: self._send_text())

        send_btn = ttk.Button(entry_frame, text="Send", command=self._send_text)
        send_btn.pack(side=tk.LEFT)

        self.mic_btn = ttk.Button(frame, text="🎤 Mic Off", command=self._toggle_mic)
        self.mic_btn.pack(anchor="e", pady=(6, 0))

    def _toggle_mic(self):
        self.mic_on = not self.mic_on
        if self.mic_btn:
            self._on_ui(self.mic_btn.config, text="🎤 Listening" if self.mic_on else "🎤 Mic Off")
        if self.mic_on:
            threading.Thread(target=self._capture_once, daemon=True).start()

    def _capture_once(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self._set_status("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            self._set_status(f"Heard: {text}")
            asyncio.run_coroutine_threadsafe(self.inbox.put(text), self.loop)
        except Exception as exc:  # Broad to keep UI alive
            self._append_log(f"Mic error: {exc}")
        finally:
            self.mic_on = False
            if self.mic_btn:
                self._on_ui(self.mic_btn.config, text="🎤 Mic Off")
            self._set_status("Ready")

    def _send_text(self):
        if not self.entry:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        asyncio.run_coroutine_threadsafe(self.inbox.put(text), self.loop)
        self._append_log(f"You: {text}")
        self._set_status("Sent")

    def _append_log(self, msg: str):
        def _do():
            if not self.log_box:
                return
            self.log_box.config(state=tk.NORMAL)
            self.log_box.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}  {msg}\n")
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)

        self._on_ui(_do)

    def _set_status(self, msg: str):
        if self.status_var:
            self._on_ui(self.status_var.set, msg)

    def start(self):
        def boot():
            self.root = tk.Tk()
            self.root.title("Kaminiel Overlay")
            self.root.attributes("-topmost", True)
            self.root.geometry("420x220")
            self.root.configure(bg="#111")
            self._build_ui()
            self._ui_ready.set()
            self.root.mainloop()

        threading.Thread(target=boot, daemon=True).start()
        self._ui_ready.wait()

    def _on_ui(self, fn, *args, **kwargs):
        if not self.root:
            return
        self.root.after(0, lambda: fn(*args, **kwargs))


class Companion:
    def __init__(self):
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY not set")
            sys.exit(1)

        self.loop = asyncio.get_event_loop()
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.overlay = OverlayApp(self.loop, self.inbox)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tts = KPipeline(lang_code="a", device=device)
        except Exception as exc:
            print(f"Kokoro init failed: {exc}")
            sys.exit(1)

        self.vts = pyvts.vts(plugin_info=PLUGIN_INFO)
        self.vts_connected = False
        self.vts_lock = asyncio.Lock()

        now = datetime.utcnow()
        self.last_user = now
        self.session_start = now

    async def connect_vts(self):
        self.overlay._append_log("🔌 Connecting to VTube Studio...")
        try:
            await self.vts.connect()

            try:
                await self.vts.request_authenticate_token()  # Reads token from file
                await self.vts.request_authenticate()        # Sends token to VTS
                self.vts_connected = True
                self.overlay._append_log("✅ Connected to VTube Studio!")
                return
            except Exception:
                self.overlay._append_log("⚠️ Auth failed. Token might be invalid. Refreshing...")
                if os.path.exists("./vts_token.txt"):
                    try:
                        os.remove("./vts_token.txt")
                    except Exception as exc:
                        self.overlay._append_log(f"Could not delete token: {exc}")

                self.overlay._append_log("👉 Check VTube Studio and click 'Allow' on the popup!")
                await self.vts.request_authenticate_token()  # Triggers popup
                await self.vts.request_authenticate()
                self.vts_connected = True
                self.overlay._append_log("✅ Re-authenticated successfully!")

        except Exception as exc:
            self.overlay._append_log(f"❌ VTS Error: {exc}")
            self.overlay._append_log("   (Is VTube Studio open? Is the API turned ON in settings?)")

    async def ask_brain(self, prompt_context: str, user_text: str) -> str:
        """Try Gemini first, then fall back to local Llama."""
        now = datetime.now(WIB_ZONE)
        part_of_day = (
            "morning"
            if 5 <= now.hour < 12
            else "afternoon"
            if 12 <= now.hour < 17
            else "evening"
            if 17 <= now.hour < 21
            else "night"
        )

        memory_lines = _active_creator_memories()
        memory_block = "\n".join(f"- {m}" for m in memory_lines or ["(none yet)"])

        full_prompt = (
            f"{PERSONA_PROMPT}\n{CREATOR_STYLE_PROMPT}\n"
            f"Time: {now.strftime('%H:%M')} ({part_of_day})\n"
            f"Active Memories:\n{memory_block}\n"
            f"Context: {prompt_context}\n"
            f"User said: {user_text}"
        )
        try:
            response = self.client.models.generate_content(model=GEMINI_MODEL, contents=full_prompt)
            if response and getattr(response, "text", None):
                return response.text.strip()
            raise RuntimeError("Empty Gemini response")
        except Exception as exc:
            self.overlay._append_log(f"⚠️ Gemini error: {exc}")
            self.overlay._append_log("🔄 Switching to Local Brain...")

        if _LOCAL_LLM_AVAILABLE:
            return await asyncio.to_thread(run_local_inference, user_text)

        return "I'm having trouble thinking (No API and No Local Model)."

    async def trigger_emotion(self, text: str) -> str:
        """Find [emotion] tag, trigger VTS, and remove it from speech."""
        tag_match = re.search(r"\[(\w+)\]", text)
        if tag_match:
            emotion = tag_match.group(1).lower()
            hotkey_id = EMOTION_HOTKEYS.get(emotion)
            if hotkey_id and self.vts_connected:
                try:
                    async with self.vts_lock:
                        await self.vts.request(self.vts.vts_request.requestTriggerHotKey(hotkeyID=hotkey_id))
                    self.overlay._append_log(f"Triggering emotion: {emotion}")
                except Exception as exc:
                    self.overlay._append_log(f"VTS Hotkey error: {exc}")
            return text.replace(tag_match.group(0), "").strip()
        return text

    def _emotion_lipsync_factor(self, text: str) -> float:
        text_lower = text.lower()
        if any(x in text_lower for x in ["happy", "excited", "yay", "cute"]):
            return 1.2
        if any(x in text_lower for x in ["sad", "tired", "down"]):
            return 0.7
        if any(x in text_lower for x in ["angry", "mad", "annoyed"]):
            return 1.0
        if any(x in text_lower for x in ["shy", "embarrassed"]):
            return 0.8
        return 1.0

    def _estimate_viseme(self, chunk: np.ndarray) -> str:
        if len(chunk) < 10:
            return "A"
        spec = np.abs(np.fft.rfft(chunk))
        low = np.mean(spec[:30])
        mid = np.mean(spec[30:120])
        high = np.mean(spec[120:300])
        if low > mid and low > high:
            return "O"
        if high > mid and high > low:
            return "I"
        return "A"

    async def speak(self, text: str):
        self.overlay._append_log(f"Kaminiel: {text}")
        await asyncio.to_thread(self._stream_audio_with_lipsync, text)

    def _stream_audio_with_lipsync(self, text: str):
        global CURRENT_LIP_SYNC_VALUE

        generator = self.tts(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, split_pattern=r"\n+")
        audio_chunks = [audio for _, _, audio in generator]
        if not audio_chunks:
            return

        try:
            full_audio = torch.cat(audio_chunks).cpu().numpy().astype(np.float32)
        except Exception:
            full_audio = np.concatenate([np.array(a, dtype=np.float32) for a in audio_chunks])

        peak = float(np.max(np.abs(full_audio))) if full_audio.size > 0 else 0.0
        if peak > 1e-6:
            full_audio = full_audio / peak
        print(
            f"[lipsync] normalized peak={peak:.6f} samples={full_audio.size} chunks={len(audio_chunks)}"
        )

        samplerate = 24000
        blocksize = 1024  # smaller block for more responsive updates

        if not hasattr(self, "auto_gain_rms"):
            self.auto_gain_rms = 1e-4

        profile = KOKORO_LIPSYNC_PROFILE.get(KOKORO_VOICE, KOKORO_LIPSYNC_PROFILE["_default"])
        voice_gain = profile.get("gain", 1.0)
        voice_bias = profile.get("bias", 0.0)

        with sd.OutputStream(samplerate=samplerate, channels=1, blocksize=blocksize) as stream:
            idx = 0
            total = len(full_audio)
            while idx < total:
                end = min(idx + blocksize, total)
                chunk = full_audio[idx:end]

                if chunk.size > 0:
                    raw_rms = float(np.sqrt(np.mean(chunk ** 2)))

                    ema_alpha = 0.05
                    self.auto_gain_rms = (1.0 - ema_alpha) * self.auto_gain_rms + ema_alpha * raw_rms
                    baseline = max(self.auto_gain_rms, 1e-6)

                    target_level = 0.25
                    adaptive_gain = target_level / baseline

                    combined_gain = min(adaptive_gain * voice_gain, 20.0)

                    calibrated = raw_rms * combined_gain + voice_bias

                    try:
                        emotion_factor = self._emotion_lipsync_factor(text)
                    except Exception:
                        emotion_factor = 1.0

                    calibrated *= emotion_factor
                    calibrated = max(0.0, min(1.0, calibrated))

                    CURRENT_LIP_SYNC_VALUE = calibrated
                    print(f"[Audio] Volume: {calibrated:.3f}")

                try:
                    stream.write(chunk.reshape(-1, 1))
                except Exception:
                    pass

                idx = end

        CURRENT_LIP_SYNC_VALUE = 0.0

    async def worry_watchdog(self):
        worry_level = 0
        while True:
            await asyncio.sleep(60)
            elapsed_mins = (datetime.utcnow() - self.last_user).total_seconds() / 60
            new_level = min(int(elapsed_mins // 15), MAX_WORRY_LEVEL)
            if new_level > worry_level:
                worry_level = new_level
                if WORRY_LEVEL_MESSAGES[worry_level]:
                    msg = random.choice(WORRY_LEVEL_MESSAGES[worry_level])
                    await self.speak(msg)
            if elapsed_mins < 1:
                worry_level = 0

    async def nudge_watchdog(self):
        while True:
            await asyncio.sleep(60)
            if datetime.utcnow() - self.session_start >= timedelta(minutes=NUDGE_AFTER_MINUTES):
                await self.inbox.put("It has been 45 minutes. Encourage a quick stretch or water break.")
                self.session_start = datetime.utcnow()

    async def _delayed_reminder(self, delay_seconds: int, topic: str):
        await asyncio.sleep(delay_seconds)
        await self.speak(f"Reminder: {topic}")
        _upsert_creator_memory("reminder", f"reminder_{topic}", f"Remind {topic} after {delay_seconds}s")

    async def handle_command(self, text: str) -> bool:
        original = text.strip()
        if not original:
            return False
        normalized = original.lower()

        if normalized.startswith("!tantrum") or normalized.startswith("throw a tantrum"):
            reason = re.sub(r"^(!tantrum|throw a tantrum)", "", original, flags=re.IGNORECASE).strip() or "no reason"
            await self.speak(f"Oh, I am going to stomp around for 5 minutes because {reason}!")
            _upsert_creator_memory("tantrum", "last_tantrum", reason)
            return True

        if normalized.startswith("!cheer"):
            topic = original[len("!cheer") :].strip() or "your task"
            template = random.choice(CHEER_MESSAGE_TEMPLATES)
            await self.speak(template.format(focus_clean=topic))
            _upsert_creator_memory("cheer", "last_cheer", topic)
            return True

        remind_match = re.match(
            r"^!remind\s+(?P<value>\d+)(?P<unit>[smhd]?)\s+(?P<topic>.+)",
            original,
            flags=re.IGNORECASE,
        )
        if remind_match:
            unit = (remind_match.group("unit") or "m").lower()
            value = int(remind_match.group("value"))
            topic = remind_match.group("topic").strip()
            seconds = value * REMIND_UNIT_MULTIPLIERS.get(unit, 60)
            asyncio.create_task(self._delayed_reminder(seconds, topic))
            await self.speak(f"Reminder set for {value}{unit} from now about {topic}.")
            _upsert_creator_memory("reminder", f"reminder_{topic}", f"{value}{unit} reminder for {topic}")
            return True

        return False

    async def generate_startup_greeting(self) -> str:
        hours_slept = _load_time_since_shutdown()
        dream = _load_dream_journal()

        if hours_slept < 5.0:
            sleep_context = (
                f"You slept for only {int(hours_slept)} hours. You feel sleepy, groggy, and are rubbing your eyes."
                " Act cute and complain gently about waking up too soon."
            )
        elif hours_slept < 24.0:
            sleep_context = (
                f"You slept for {int(hours_slept)} hours. You feel completely refreshed, energetic, and happy"
                " to see your creator again."
            )
        else:
            days = max(1, int(hours_slept // 24))
            sleep_context = (
                f"You have been gone for {days} days. You missed your creator deeply and feel lonely and clingy."
                " Express how hard the wait was."
            )

        dream_context = ""
        if dream:
            dream_context = f"You also remember this from before you slept: '{dream}'. Mention this memory."

        prompt = (
            "You are waking up (starting up) on the desktop.\n"
            f"{sleep_context}\n"
            f"{dream_context}\n"
            "Greet your creator, Kamitchi. Keep it to 1-2 warm sentences."
        )

        greeting = await self.ask_brain("System Startup Event", prompt)

        if hours_slept < 5:
            greeting += " [shy]"
        elif hours_slept > 24:
            greeting += " [sad]"
        else:
            greeting += " [happy]"

        return greeting

    async def lip_sync_loop(self):
        """Continuously sends MouthOpen via SetParameterValue, exactly like test.py."""
        print("[LipSync] Loop started.")
        while True:
            await asyncio.sleep(0.03)  # ~30 FPS
            if not self.vts_connected:
                continue

            val = 0.0
            if isinstance(CURRENT_LIP_SYNC_VALUE, tuple):
                val = CURRENT_LIP_SYNC_VALUE[0]
            else:
                val = CURRENT_LIP_SYNC_VALUE

            target_value = max(0.0, min(1.0, val * 3.0))

            try:
                async with self.vts_lock:
                    await self.vts.request(
                        self.vts.vts_request.requestSetParameterValue(
                            parameter="MouthOpen",
                            value=float(target_value),
                        )
                    )
            except Exception as e:
                print(f"[LipSync Error] {e}")
                await asyncio.sleep(1)

    async def brain(self):
        await self.connect_vts()
        self.overlay.start()
        self.overlay._append_log("Kaminiel is waking up...")
        greeting = await self.generate_startup_greeting()
        clean_greeting = await self.trigger_emotion(greeting)
        await self.speak(clean_greeting)
        _save_dream_journal("")

        asyncio.create_task(self.worry_watchdog())
        asyncio.create_task(self.nudge_watchdog())
        asyncio.create_task(self.lip_sync_loop())

        self.overlay._append_log("Ready for input.")

        while True:
            text = await self.inbox.get()
            self.last_user = datetime.utcnow()
            if await self.handle_command(text):
                continue
            if text.lower() in {"goodbye", "shutdown", "exit", "sleep"}:
                _save_last_shutdown_time()
                await self.speak("Goodnight, Kamitchi. [sleepy]")
                break

            raw_reply = await self.ask_brain("User is talking directly.", text)
            clean_reply = await self.trigger_emotion(raw_reply)
            await self.speak(clean_reply)

        if self.vts_connected:
            await self.vts.close()
        self.overlay._append_log("Stopped.")

    def run(self):
        self.loop.run_until_complete(self.brain())


def main():
    Companion().run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
