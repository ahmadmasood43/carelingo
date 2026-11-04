"""Synthetic dictation audio with Piper, for end-to-end tests without recordings.

Synthetic speech is cleaner and more regular than a nurse dictating on a ward, so
speech-recognition results on it are a best case. Background noise is added to make
the audio a little less ideal, not to model a real ward.

Needs the optional `tts` dependency group (`uv sync --group tts`).
"""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Voice:
    name: str  # Piper model file stem, e.g. "en_GB-alba-medium"
    speaker: int | None = None
    label: str = ""


# One native and four non-native English voices (L2-ARCTIC speakers whose first
# languages are Arabic, Mandarin, Hindi and Spanish), and three Polish voices.
VOICES = {
    "en": [
        Voice("en_GB-alba-medium", label="native"),
        Voice("en_US-l2arctic-medium", 21, "L1 Arabic"),
        Voice("en_US-l2arctic-medium", 20, "L1 Mandarin"),
        Voice("en_US-l2arctic-medium", 10, "L1 Hindi"),
        Voice("en_US-l2arctic-medium", 22, "L1 Spanish"),
    ],
    "pl": [
        Voice("pl_PL-gosia-medium", label="native"),
        Voice("pl_PL-darkman-medium", label="native"),
        Voice("pl_PL-mc_speech-medium", label="native"),
    ],
}


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with a 1/f power spectrum, closer to room noise than white noise."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequencies = np.arange(len(spectrum))
    frequencies[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(frequencies), n)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = pink_noise(len(audio), rng)
    signal_power = np.mean(audio**2)
    noise *= np.sqrt(signal_power / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    mixed = audio + noise
    return mixed / max(1.0, np.max(np.abs(mixed)))


class Synthesizer:
    def __init__(self, voice_dir: Path):
        self._voice_dir = voice_dir
        self._loaded = {}

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """Audio as float32 in [-1, 1], and its sample rate."""
        from piper import PiperVoice, SynthesisConfig

        if voice.name not in self._loaded:
            self._loaded[voice.name] = PiperVoice.load(self._voice_dir / f"{voice.name}.onnx")
        model = self._loaded[voice.name]
        chunks = model.synthesize(text, SynthesisConfig(speaker_id=voice.speaker))
        audio = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        return audio.astype(np.float32) / 32768, model.config.sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
