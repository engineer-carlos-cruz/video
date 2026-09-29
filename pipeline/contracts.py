"""Stable contracts between pipeline steps.

These dataclasses are the *only* thing the steps agree on. A step's internal
implementation (library, engine, even language) may change as long as it keeps
producing and consuming these types.

Each dataclass exposes ``is_empty()`` so the orchestrator can detect a step that
finished without usable output in a uniform way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Source = Literal["youtube", "spotify"]


@dataclass(frozen=True)
class Metadata:
    """Travels along the whole pipeline without coupling any step."""

    url: str
    source: Source
    title: str
    artist: str
    album: str | None = None
    source_id: str | None = None

    def is_empty(self) -> bool:
        return not self.title.strip() or not self.artist.strip()


@dataclass(frozen=True)
class DownloadedAudio:
    """Step 1 output: a lossless PCM ``.wav`` plus its metadata."""

    file: Path
    metadata: Metadata

    def is_empty(self) -> bool:
        return not self.file.exists() or self.file.stat().st_size == 0


@dataclass(frozen=True)
class CleanedAudio:
    """Step 2 output: denoised ``.wav`` with the original metadata preserved."""

    file: Path
    metadata: Metadata

    def is_empty(self) -> bool:
        return not self.file.exists() or self.file.stat().st_size == 0


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    """Step 3 output: text, language and per-word timings."""

    text: str
    language: str
    words: list[Word] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.text.strip() or not self.words


@dataclass(frozen=True)
class Phrase:
    id: str
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Phrases:
    """Step 4 output: one timed entry per phrase, first occurrence."""

    phrases: list[Phrase] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.phrases


@dataclass(frozen=True)
class Fragment:
    id: str
    file: Path
    start: float
    end: float


@dataclass(frozen=True)
class Fragments:
    """Step 5 output: one ``.wav`` per phrase, ~100 ms of margin on both sides."""

    fragments: list[Fragment] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.fragments


@dataclass(frozen=True)
class Card:
    """Consumer-agnostic view of a video: audio + EN + ES + timings."""

    id: str
    en: str
    es: str
    audio: Path
    start: float
    end: float


@dataclass(frozen=True)
class Dataset:
    """Step 6 output. Not oriented towards any final consumer."""

    cards: list[Card] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.cards


@dataclass(frozen=True)
class AnkiPackage:
    """Step 7 output: a ready-to-import ``.apkg``."""

    apkg_path: Path

    def is_empty(self) -> bool:
        return not self.apkg_path.exists() or self.apkg_path.stat().st_size == 0
