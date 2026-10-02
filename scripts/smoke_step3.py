#!/usr/bin/env python3
"""Smoke test for step 3.

Offline and deterministic: the fixture is synthesised speech (ffmpeg's ``flite``
voice) resampled to the exact format step 2 hands over — 48 kHz stereo 16-bit
wav — so this also covers the decoding path the pipeline actually uses.

The transcription thresholds are expressed as token agreement rather than exact
strings, because the accuracy of a synthetic voice varies with the flite build;
the first time this ran on this machine the fixture was transcribed at 100% of
its tokens, so 0.6 leaves room for a different ffmpeg without hiding a silent
regression (a broken word-timestamp run lands far below it).

    python scripts/smoke_step3.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import CleanedAudio, Metadata  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.media import probe_wav  # noqa: E402
from pipeline.steps.step3_transcribe import Step3Transcribe  # noqa: E402

SPEECH_LINES = (
    "the quick brown fox jumps over the lazy dog",
    "i have been learning english for two years",
    "she is going to the store right now",
)

MIN_TOKEN_AGREEMENT = 0.6
MAX_REAL_TIME_FACTOR = 10.0

METADATA = Metadata(
    url="https://www.youtube.com/watch?v=fixture",
    source="youtube",
    title="Fixture Title",
    artist="Fixture Artist",
    album=None,
    source_id="fixture",
)


def flite_source() -> str:
    """The speech source, as one argv entry so the filtergraph parser is happy."""
    joined = " ".join(SPEECH_LINES)
    return f"flite=text='{joined}':voice=slt"


def build_fixture(path: Path, source: str = flite_source()) -> Path:
    """Speech in the same wav layout step 2 produces."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", source,
            "-af", "aresample=48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    return path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def agreement(expected: list[str], heard: list[str]) -> float:
    """Fraction of expected tokens present in the transcription, in order.

    A plain multiset overlap: the whole point of the step is that the words come
    out in the order they are spoken, but flite/Whisper disagreeing on a couple
    of tokens should not fail the run.
    """
    remaining = list(heard)
    matched = 0
    for token in expected:
        if token in remaining:
            remaining.remove(token)
            matched += 1
    return matched / len(expected) if expected else 0.0


def transcribe(fixture: Path, **kwargs) -> tuple:
    cleaned = CleanedAudio(file=fixture, metadata=METADATA)
    step = Step3Transcribe(**kwargs)
    started = time.monotonic()
    transcript = step.run(cleaned)
    return transcript, time.monotonic() - started, step


def check_transcription(workspace: Path) -> None:
    print("== transcription ==")
    fixture = build_fixture(workspace / "speech.wav")
    transcript, _, _ = transcribe(fixture)

    expected = tokenize(" ".join(SPEECH_LINES))
    heard = tokenize(transcript.text)
    score = agreement(expected, heard)
    print(f"  language: {transcript.language}")
    print(f"  text: {transcript.text}")
    print(f"  token agreement: {score:.0%} ({len(transcript.words)} words timed)")
    assert transcript.language == "en", transcript.language
    assert score >= MIN_TOKEN_AGREEMENT, f"only {score:.0%} of the fixture was recognised"


def check_word_timestamps(workspace: Path) -> None:
    print("== word timestamps ==")
    fixture = build_fixture(workspace / "timings.wav")
    duration = probe_wav(fixture).duration
    transcript, _, _ = transcribe(fixture)

    assert transcript.words, "no word came with a timestamp"
    previous_end = -1.0
    for word in transcript.words:
        assert word.end >= word.start, word
        assert word.start >= previous_end, f"{word} overlaps the previous word"
        assert word.start <= duration, f"{word} starts after the end of the audio"
        previous_end = word.end
    first, last = transcript.words[0], transcript.words[-1]
    print(f"  {len(transcript.words)} words, monotonic, within {duration:.2f}s")
    print(f"  first={first.word!r}@{first.start:.2f}s last={last.word!r}@{last.end:.2f}s")

    # Every word must carry a real duration, otherwise step 4 cannot align.
    shortest = min(w.end - w.start for w in transcript.words)
    longest = max(w.end - w.start for w in transcript.words)
    print(f"  word duration: {shortest:.2f}s .. {longest:.2f}s")
    assert longest < 5.0, f"a word lasts {longest:.2f}s, the timings are not aligned"


def check_vad(workspace: Path) -> None:
    print("== vad drops non-speech ==")
    silence = workspace / "silence.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=5",
            "-c:a", "pcm_s16le", str(silence),
        ],
        check=True,
    )
    try:
        Step3Transcribe().run(CleanedAudio(file=silence, metadata=METADATA))
    except StepOutputError as exc:
        assert "no speech" in str(exc), exc
        print("  silent audio -> StepOutputError")
    else:
        raise AssertionError("silent audio must not produce a transcript")


def check_speed(workspace: Path) -> None:
    print("== speed on cpu ==")
    fixture = build_fixture(workspace / "speed.wav")
    duration = probe_wav(fixture).duration
    transcript, elapsed, _ = transcribe(fixture)
    rtf = elapsed / duration
    print(f"  {duration:.2f}s of audio in {elapsed:.2f}s (real time factor {rtf:.2f}x)")
    assert transcript.words
    assert rtf < MAX_REAL_TIME_FACTOR, f"took {rtf:.1f}x real time, the model is too slow"


def check_model_reuse(workspace: Path) -> None:
    print("== the model is loaded once ==")
    fixture = build_fixture(workspace / "reuse.wav")
    step = Step3Transcribe()
    assert step._whisper is None
    step.run(CleanedAudio(file=fixture, metadata=METADATA))
    loaded = step._whisper
    assert loaded is not None
    step.run(CleanedAudio(file=fixture, metadata=METADATA))
    assert step._whisper is loaded, "the model was reloaded between runs"
    print("  same WhisperModel instance across runs")
    print("  ok")


def check_errors(workspace: Path) -> None:
    print("== errors ==")
    step = Step3Transcribe()

    try:
        step.run(CleanedAudio(file=workspace / "nope.wav", metadata=METADATA))
    except StepOutputError as exc:
        assert "empty input" in str(exc), exc
    print("  missing input -> StepOutputError")

    broken = workspace / "broken.wav"
    broken.write_bytes(b"RIFF not really a wav")
    try:
        step.run(CleanedAudio(file=broken, metadata=METADATA))
    except StepOutputError as exc:
        assert "WAV" in str(exc), exc
    print("  corrupt input -> StepOutputError")

    missing_model = Step3Transcribe(model=workspace / "no-such-model")
    fixture = build_fixture(workspace / "badmodel.wav")
    try:
        missing_model.run(CleanedAudio(file=fixture, metadata=METADATA))
    except StepOutputError as exc:
        assert "Whisper model" in str(exc), exc
    print("  missing model -> StepOutputError")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        check_transcription(workspace)
        check_word_timestamps(workspace)
        check_vad(workspace)
        check_speed(workspace)
        check_model_reuse(workspace)
        check_errors(workspace)
    print("\nall good")
