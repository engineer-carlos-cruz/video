"""Step 1 — download the audio of a YouTube or Spotify URL.

Produces a lossless PCM ``.wav`` (the intermediate format for the rest of the
pipeline) plus the metadata needed downstream, in particular the audio name.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from pipeline.contracts import DownloadedAudio, Metadata
from pipeline.errors import StepOutputError, UnsupportedURLError

YOUTUBE_PATTERNS = (
    r"(^|\.)youtube\.com$",
    r"(^|\.)youtu\.be$",
    r"(^|\.)music\.youtube\.com$",
)
SPOTIFY_PATTERNS = (r"(^|\.)spotify\.com$",)

DEFAULT_OUTPUT_DIR = Path("output")

_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def slugify(value: str, max_length: int = 80) -> str:
    """Return a safe, readable filename fragment."""
    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:max_length] or "audio"


class AudioSource(ABC):
    """Internal interface shared by the supported platforms."""

    name: str

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)

    @abstractmethod
    def fetch(self, url: str) -> tuple[Path, Metadata]:
        """Download ``url`` and return the wav path plus its metadata."""

    @staticmethod
    def verify_wav(path: Path) -> None:
        """Sanity-check the produced file, raising on anything unusable."""
        if not path.exists() or path.stat().st_size == 0:
            raise StepOutputError(f"The downloaded file is missing or empty: {path}")
        try:
            with wave.open(str(path), "rb") as handle:
                channels, rate, frames = (
                    handle.getnchannels(),
                    handle.getframerate(),
                    handle.getnframes(),
                )
        except wave.Error as exc:
            raise StepOutputError(f"{path} is not a readable WAV file: {exc}") from exc
        if frames == 0 or rate == 0:
            raise StepOutputError(f"{path} contains no audio samples")

    def _require_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise StepOutputError(
                "ffmpeg was not found in PATH; it is required to extract audio"
            )


class YoutubeSource(AudioSource):
    """Downloads the best available audio track through ``yt-dlp``."""

    name = "youtube"

    def fetch(self, url: str) -> tuple[Path, Metadata]:
        self._require_ffmpeg()
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise StepOutputError("yt-dlp is not installed in this environment") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "quiet": True,
            "noprogress": True,
        }

        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise StepOutputError(f"yt-dlp could not download {url}: {exc}") from exc

        if not info:
            raise StepOutputError(f"yt-dlp returned no information for {url}")

        video_id = info.get("id") or slugify(info.get("title", "audio"))
        wav_path = self.output_dir / f"{video_id}.wav"
        self.verify_wav(wav_path)

        metadata = Metadata(
            url=url,
            source="youtube",
            title=(info.get("title") or "").strip(),
            artist=(info.get("artist") or info.get("uploader") or info.get("channel") or "").strip(),
            album=info.get("album"),
            source_id=info.get("id"),
        )
        if metadata.is_empty():
            raise StepOutputError(f"Incomplete metadata for {url}: {metadata}")
        return wav_path, metadata


class SpotifySource(AudioSource):
    """Resolves the track metadata and the audio through the ``spotdl`` CLI."""

    name = "spotify"

    def _spotdl_command(self) -> list[str]:
        executable = shutil.which("spotdl")
        if executable:
            return [executable]
        # Installed in the venv but its bin dir is not on PATH.
        candidate = Path(sys.executable).parent / "spotdl"
        if candidate.exists():
            return [str(candidate)]
        return [sys.executable, "-m", "spotdl"]

    def _run(self, args: list[str], capture_stdout: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args,
                check=True,
                text=True,
                stdout=subprocess.PIPE if capture_stdout else None,
                stderr=subprocess.PIPE if capture_stdout else None,
            )
        except FileNotFoundError as exc:
            raise StepOutputError("spotdl is not installed in this environment") from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or "").strip().splitlines()
            reason = details[-1] if details else f"exit code {exc.returncode}"
            raise StepOutputError(f"spotdl failed: {reason}") from exc

    def _song_metadata(self, url: str) -> dict:
        result = self._run(
            [*self._spotdl_command(), "save", url, "--save-file", "-"],
            capture_stdout=True,
        )
        payload = (result.stdout or "").strip()
        # spotdl prefixes its output with a "Processing query: ..." line.
        start = min(
            (i for i in (payload.find("["), payload.find("{")) if i != -1),
            default=-1,
        )
        if start == -1:
            raise StepOutputError(f"spotdl returned no metadata for {url}")
        try:
            data = json.loads(payload[start:])
        except json.JSONDecodeError as exc:
            raise StepOutputError(f"spotdl returned invalid metadata for {url}") from exc
        songs = data if isinstance(data, list) else [data]
        if not songs:
            raise StepOutputError(f"spotdl found no song for {url}")
        return songs[0]

    @staticmethod
    def _artist(song: dict) -> str:
        artists = song.get("artists") or []
        names = []
        for artist in artists:
            if isinstance(artist, str):
                names.append(artist)
            elif isinstance(artist, dict) and artist.get("name"):
                names.append(artist["name"])
        if not names and song.get("artist"):
            names = [song["artist"]]
        return ", ".join(names)

    @staticmethod
    def _album(song: dict) -> str | None:
        album = song.get("album_name")
        if isinstance(album, str) and album:
            return album
        nested = song.get("album")
        if isinstance(nested, dict):
            return nested.get("name")
        if isinstance(nested, str):
            return nested
        return None

    def fetch(self, url: str) -> tuple[Path, Metadata]:
        self._require_ffmpeg()
        song = self._song_metadata(url)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        template = str(self.output_dir / "{title}.{output-ext}")
        self._run(
            [
                *self._spotdl_command(),
                "download",
                url,
                "--format",
                "wav",
                "--output",
                template,
                "--overwrite",
                "force",
                "--print-errors",
            ]
        )

        title = (song.get("name") or "").strip()
        wav_path = self.output_dir / f"{slugify(title)}.wav"
        if not wav_path.exists():
            candidates = sorted(self.output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise StepOutputError(f"spotdl produced no wav file in {self.output_dir}")
            wav_path = candidates[-1]
        self.verify_wav(wav_path)

        metadata = Metadata(
            url=url,
            source="spotify",
            title=title,
            artist=self._artist(song),
            album=self._album(song),
            source_id=song.get("song_id") or song.get("track_id") or song.get("uri"),
        )
        if metadata.is_empty():
            raise StepOutputError(f"Incomplete metadata for {url}: {metadata}")
        return wav_path, metadata


def detect_platform(url: str) -> str:
    """Return ``"youtube"`` or ``"spotify"`` based on the URL host."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        raise UnsupportedURLError(f"Could not parse a host out of {url!r}")
    for pattern in YOUTUBE_PATTERNS:
        if re.search(pattern, host):
            return "youtube"
    for pattern in SPOTIFY_PATTERNS:
        if re.search(pattern, host):
            return "spotify"
    raise UnsupportedURLError(
        f"Unsupported URL {url!r}; only YouTube and Spotify are supported"
    )


class Step1Download:
    """Downloads the audio for a YouTube or Spotify URL."""

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self._sources: dict[str, AudioSource] = {
            "youtube": YoutubeSource(self.output_dir),
            "spotify": SpotifySource(self.output_dir),
        }

    def run(self, url: str) -> DownloadedAudio:
        source = self._sources[detect_platform(url)]
        try:
            file, metadata = source.fetch(url)
        except StepOutputError:
            raise
        except Exception as exc:
            raise StepOutputError(f"Step 1 failed for {url}: {exc}") from exc

        result = DownloadedAudio(file=file, metadata=metadata)
        if result.is_empty():
            raise StepOutputError(f"Step 1 produced no usable output for {url}")
        return result
