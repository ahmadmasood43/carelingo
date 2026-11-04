"""Speech recognition with faster-whisper, and word error rate.

Needs the optional `asr` dependency group (`uv sync --group asr`).
"""

import ctypes
import re
import site
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def _preload_cuda_libraries() -> None:
    # CTranslate2 dlopens cuBLAS and cuDNN by soname. The pip-installed copies are not
    # on the loader path, and LD_LIBRARY_PATH cannot be changed for a running process,
    # so load them explicitly before CTranslate2 needs them.
    for root in map(Path, site.getsitepackages()):
        for pattern in ("nvidia/cublas/lib/libcublas*.so.*", "nvidia/cudnn/lib/libcudnn*.so.*"):
            for library in sorted(root.glob(pattern)):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def load_audio(path: Path) -> np.ndarray:
    """Any audio file as 16 kHz mono float32, decoded by ffmpeg.

    Decoding outside faster-whisper avoids its PyAV dependency, whose recent versions
    are incompatible with faster-whisper 1.2.1.
    """
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path)]
        + ["-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    language_probability: float
    audio_s: float
    latency_s: float


class Transcriber:
    # Pascal GPUs have no fast float16; int8 is the fastest supported compute type.
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", compute: str = "int8"):
        if device == "cuda":
            _preload_cuda_libraries()
        from faster_whisper import WhisperModel

        self.model_name = model
        self._model = WhisperModel(model, device=device, compute_type=compute)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio, language=language, beam_size=5, vad_filter=True, condition_on_previous_text=False
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return Transcript(
            text,
            info.language,
            info.language_probability,
            len(audio) / SAMPLE_RATE,
            time.perf_counter() - started,
        )


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus WER after lowercasing and removing punctuation.

    Numbers are not normalised: a reference "one hundred" against "100" counts as
    errors, which overstates WER on number-heavy text.
    """
    import jiwer

    return jiwer.wer([_normalize(r) for r in references], [_normalize(h) for h in hypotheses])
