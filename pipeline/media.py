"""Shared audio helpers.

Not a coupling point between steps: this is a thin wrapper around the stdlib
``wave`` module so that every step validates and inspects wav files the same way.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from pipeline.errors import StepOutputError


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_bits: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def probe_wav(path: Path) -> WavInfo:
    """Return the format of ``path``, raising if it is not a usable wav."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise StepOutputError(f"The file is missing or empty: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            info = WavInfo(
                sample_rate=handle.getframerate(),
                channels=handle.getnchannels(),
                sample_bits=handle.getsampwidth() * 8,
                frames=handle.getnframes(),
            )
    except wave.Error as exc:
        raise StepOutputError(f"{path} is not a readable WAV file: {exc}") from exc
    if info.frames == 0 or info.sample_rate == 0:
        raise StepOutputError(f"{path} contains no audio samples")
    return info


def verify_wav(path: Path) -> WavInfo:
    """Probe ``path`` and fail on anything unusable."""
    return probe_wav(path)