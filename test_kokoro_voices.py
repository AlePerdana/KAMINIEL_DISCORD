"""Quick helper script to audition Kokoro voices with a sample sentence."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from kokoro import KPipeline

SAMPLE_RATE = 24_000
SAMPLE_TEXT = "A brown fox named Kaminiel vaults over a lazy robot guardian."
SAMPLE_TEXT_JP = "こんにちは、これは日本語の音声テストです。"
DEFAULT_SPEED = float(os.getenv("KOKORO_TEST_SPEED", "1.1"))

DEFAULT_VOICES: tuple[str, ...] = (
# US English (female)
    "af_bella",
    "af_sarah",
    "af_nicole",
    "af_alloy",
    "af_aoede",
    "af_nova",
    "af_river",
    "af_sky",
    # US English (male)
    "am_adam",
    "am_liam",
    # British English (Corrected)
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    # --- Japanese speakers (Corrected) ---
    # These are the actual voice IDs from the model
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
)


def _voice_list_from_env() -> Iterable[str]:
    override = os.getenv("KOKORO_TEST_VOICES", "").strip()
    if not override:
        return DEFAULT_VOICES
    return [voice.strip() for voice in override.split(",") if voice.strip()]


def synthesize_voice_sample(pipeline: KPipeline, voice: str, out_dir: Path, text: str, speed: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_voice = voice.replace("/", "_")
    output_path = out_dir / f"{safe_voice}.wav"

    audio_chunks: list[torch.Tensor] = []
    for _, _, audio_tensor in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio_tensor)

    if not audio_chunks:
        raise RuntimeError(f"No audio returned for voice '{voice}'.")

    try:
        final_audio = torch.cat(audio_chunks).cpu().numpy()
    except Exception:
        final_audio = torch.cat([chunk.cpu() for chunk in audio_chunks]).numpy()

    sf.write(output_path, final_audio, SAMPLE_RATE)
    return output_path


def main() -> None:
    lang_code = os.getenv("KOKORO_LANG_CODE", "a")
    output_folder = Path(os.getenv("KOKORO_TEST_OUTPUT", "kokoro_voice_samples"))
    voices = list(_voice_list_from_env())

    if not voices:
        raise SystemExit("No voices to test. Set KOKORO_TEST_VOICES or use the defaults.")

    print(f"Loading Kokoro pipeline (lang_code='{lang_code}')...")
    pipeline = KPipeline(lang_code=lang_code)

    print(f"Generating samples for {len(voices)} voice(s) into '{output_folder}'")
    for voice in voices:
        # --- Start of fix ---
        # Check the voice prefix and select the correct text
        if voice.startswith("jp_"):
            text_to_use = SAMPLE_TEXT_JP
        else:
            text_to_use = SAMPLE_TEXT
        # --- End of fix ---

        try:
            # --- Update this line to use the new `text_to_use` variable ---
            audio_path = synthesize_voice_sample(
                pipeline, voice, output_folder, text_to_use, DEFAULT_SPEED
            )
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {voice}: {exc}")
            continue
        print(f"✓ {voice} -> {audio_path}")


if __name__ == "__main__":
    main()
