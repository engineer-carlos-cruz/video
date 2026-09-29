#!/usr/bin/env python3
"""Smoke test for step 1.

Offline: builds a synthetic wav with ffmpeg and runs the real ``YoutubeSource``
against a fake ``yt_dlp.YoutubeDL``, so the contract, the metadata mapping and
the wav validation are checked without touching the network.

Online (optional, pass a URL):

    python scripts/smoke_step1.py "https://www.youtube.com/watch?v=..."
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import DownloadedAudio  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.steps.step1_download import (  # noqa: E402
    Step1Download,
    detect_platform,
    slugify,
)

FAKE_INFO = {
    "id": "abc123",
    "title": "Sample Video Title",
    "uploader": "Sample Channel",
    "album": None,
}


def make_sine(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ac", "1", "-ar", "16000", str(path),
        ],
        check=True,
    )
    return path


class FakeYDL:
    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        assert "bestaudio/best" in self.options["format"]
        assert any(
            pp.get("preferredcodec") == "wav" for pp in self.options["postprocessors"]
        )
        return dict(FAKE_INFO)


def check_offline() -> None:
    print("== offline ==")
    import yt_dlp

    original = yt_dlp.YoutubeDL
    yt_dlp.YoutubeDL = FakeYDL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            make_sine(workdir / "abc123.wav")
            step = Step1Download(workdir)
            out = step.run("https://www.youtube.com/watch?v=abc123")

            assert isinstance(out, DownloadedAudio), out
            assert out.file.name == "abc123.wav", out.file
            assert out.file.exists() and not out.is_empty()
            assert out.metadata.source == "youtube"
            assert out.metadata.title == "Sample Video Title", out.metadata
            assert out.metadata.artist == "Sample Channel", out.metadata
            assert out.metadata.source_id == "abc123"
        print(f"  wav:    {out.file.name}")
        print(f"  title:  {out.metadata.title}")
        print(f"  artist: {out.metadata.artist}")
        print("  ok")
    finally:
        yt_dlp.YoutubeDL = original

    print("== failure wrapping ==")
    class BoomYDL(FakeYDL):
        def extract_info(self, url, download=True):
            raise RuntimeError("403 Forbidden")

    yt_dlp.YoutubeDL = BoomYDL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                Step1Download(Path(tmp)).run("https://youtu.be/x")
            except StepOutputError as exc:
                assert "403" in str(exc), exc
            else:
                raise AssertionError("a download failure must raise StepOutputError")
    finally:
        yt_dlp.YoutubeDL = original
    print("  ok")

    print("== platform detection ==")
    assert detect_platform("https://youtu.be/dQw4w9WgXcQ") == "youtube"
    assert detect_platform("https://m.youtube.com/watch?v=x") == "youtube"
    assert detect_platform("https://www.youtube.com/watch?v=x") == "youtube"
    assert detect_platform("https://open.spotify.com/track/abc") == "spotify"
    for bad in ("https://vimeo.com/1", "not-a-url", ""):
        try:
            detect_platform(bad)
        except StepOutputError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")
    print("  ok")

    print("== slugify ==")
    assert slugify('A/B: "C" <D>') == "A B C D", slugify('A/B: "C" <D>')
    assert slugify("   ") == "audio"
    print("  ok")


def check_online(url: str) -> None:
    print(f"== online ({url}) ==")
    out = Step1Download().run(url)
    assert out.file.suffix == ".wav", out.file
    print(f"  wav:    {out.file}  ({out.file.stat().st_size / 1e6:.1f} MB)")
    print(f"  source: {out.metadata.source}")
    print(f"  title:  {out.metadata.title}")
    print(f"  artist: {out.metadata.artist}")
    print(f"  album:  {out.metadata.album}")
    print("  ok")


if __name__ == "__main__":
    check_offline()
    if len(sys.argv) > 1:
        check_online(sys.argv[1])
    print("\nall good")
