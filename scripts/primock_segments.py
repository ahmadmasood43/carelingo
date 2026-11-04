"""Cut PriMock57 doctor recordings into utterances and write a manifest for `carelingo asr`.

Usage: primock_segments.py DIR, where DIR holds subset.txt (consultation names),
transcripts/*_doctor.TextGrid and wav/*_doctor.wav from the PriMock57 repository.
"""

import re
import subprocess
import sys
from pathlib import Path

INTERVAL = re.compile(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"')
FILLERS = re.compile(r"\b(um+|uh+|erm+|er|hmm+|mm+|mhm|ah)\b[,.]?\s*", re.IGNORECASE)
MIN_SECONDS = 1.5
MIN_WORDS = 3


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return FILLERS.sub("", text).strip()


def main() -> None:
    root = Path(sys.argv[1])
    out = root / "utterances"
    out.mkdir(exist_ok=True)
    rows = []
    for name in (root / "subset.txt").read_text().split():
        grid = (root / "transcripts" / f"{name}_doctor.TextGrid").read_text()
        for i, (start, end, text) in enumerate(INTERVAL.findall(grid)):
            start, end = float(start), float(end)
            if not text.strip() or "<UNIN" in text or end - start < MIN_SECONDS:
                continue
            reference = clean(text)
            if len(reference.split()) < MIN_WORDS:
                continue
            utterance = f"{name}-{i:03d}"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
                + ["-ss", f"{start}", "-to", f"{end}"]
                + ["-i", str(root / "wav" / f"{name}_doctor.wav"), str(out / f"{utterance}.wav")],
                check=True,
            )
            rows.append(f"{utterance}\t{utterance}.wav\t{reference}")
    (root / "manifest-doctor.tsv").write_text("\n".join(rows) + "\n")
    print(len(rows), "utterances")


if __name__ == "__main__":
    main()
