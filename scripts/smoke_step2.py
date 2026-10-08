#!/usr/bin/env python3
"""Smoke test for step 2.

Offline and deterministic: the fixture is real synthesised speech (ffmpeg's
``flite`` voice) mixed with brown noise, so SNR can be measured on a noise-only
segment and on the speech segment.

Thresholds were measured on this machine with the fixtures below; they are
deliberately loose so a different ffmpeg build does not cause spurious failures,
but tight enough to catch a silent regression.

    python scripts/smoke_step2.py
"""

from __future__ import annotations

import io
import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import DownloadedAudio, Metadata  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.media import probe_wav  # noqa: E402
from pipeline.steps import step2_denoise  # noqa: E402
from pipeline.steps.step2_denoise import Step2Denoise  # noqa: E402

SPEECH_TEXT = "the quick brown fox jumps over the lazy dog"
NOISE_SECONDS = 2.0

# fixture layout: [noise only][speech over noise]
NOISE_WINDOW = (0.2, 1.8)
SPEECH_WINDOW = (2.3, 4.6)

MIN_SNR_GAIN_DB = 10.0  # measured: 8.57 dB -> 39.0 dB
MAX_DURATION_DRIFT = 1.0 / 48000

METADATA = Metadata(
    url="https://www.youtube.com/watch?v=fixture",
    source="youtube",
    title="Fixture Title",
    artist="Fixture Artist",
    album=None,
    source_id="fixture",
)


def flite_source() -> str:
    """The speech source. Quotes survive the filtergraph parser because this
    is passed as a single argv entry, not through a shell."""
    return f"flite=text='{SPEECH_TEXT}':voice=slt"


def build_fixture(path: Path) -> Path:
    """Noise-only segment followed by speech over the same noise."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"anoisesrc=color=brown:amplitude=0.15:duration={NOISE_SECONDS}",
            "-f", "lavfi", "-i", flite_source(),
            "-filter_complex",
            "[1]aformat=sample_fmts=fltp:channel_layouts=mono[sp];"
            "[0]aformat=sample_fmts=fltp:channel_layouts=mono,volume=0.5[nz];"
            "[nz][sp]concat=n=2:v=0:a=1,"
            "aresample=48000,aformat=channel_layouts=stereo",
            str(path),
        ],
        check=True,
    )
    return path


def read_samples(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return list(struct.unpack("<%dh" % (len(raw) // 2), raw)), handle.getframerate()


def rms(samples: list[int], rate: int, window: tuple[float, float]) -> float:
    start, end = int(window[0] * rate), int(window[1] * rate)
    chunk = samples[start:end]
    if not chunk:
        raise AssertionError(f"empty window {window} in a {len(samples)} sample file")
    return math.sqrt(sum(v * v for v in chunk) / len(chunk))


def snr_db(samples: list[int], rate: int) -> float:
    noise = max(rms(samples, rate, NOISE_WINDOW), 1.0)
    speech = rms(samples, rate, SPEECH_WINDOW)
    return 20 * math.log10(speech / noise)


def check_snr(workspace: Path) -> None:
    print("== snr improvement ==")
    fixture = build_fixture(workspace / "fixture.wav")
    noisy = DownloadedAudio(file=fixture, metadata=METADATA)

    clean = Step2Denoise(output_dir=workspace).run(noisy)

    before_samples, before_rate = read_samples(fixture)
    after_samples, after_rate = read_samples(clean.file)
    before, after = snr_db(before_samples, before_rate), snr_db(after_samples, after_rate)
    gain = after - before
    print(f"  snr: {before:.2f} dB -> {after:.2f} dB  (gain {gain:+.2f} dB)")
    assert gain > MIN_SNR_GAIN_DB, f"denoising only improved snr by {gain:.2f} dB"
    print("  ok")


def check_duration(workspace: Path) -> None:
    print("== duration is preserved ==")
    fixture = build_fixture(workspace / "duration.wav")
    clean = Step2Denoise(output_dir=workspace).run(
        DownloadedAudio(file=fixture, metadata=METADATA)
    )
    before = probe_wav(fixture)
    after = probe_wav(clean.file)
    drift = abs(after.duration - before.duration)
    print(f"  {before.duration:.6f} s -> {after.duration:.6f} s  (drift {drift * 1000:.4f} ms)")
    assert drift <= MAX_DURATION_DRIFT, f"duration drifted by {drift * 1000:.4f} ms"
    print("  ok")


def check_sample_rates(workspace: Path) -> None:
    print("== the duration holds whatever the input's sample rate ==")
    # Regression. Every other fixture here is built with aresample=48000, which
    # hid this: atrim=end_sample counts samples in the *output* stream (48 kHz),
    # while before.frames counts them in the input. Step 1 keeps the source rate
    # (44.1 kHz is typical for YouTube and Spotify), so a 3 s track came out
    # 2.756 s and the step refused it. Measured truncations: 44100 -> 2.756 s,
    # 22050 -> 1.378 s, 16000 -> 1.000 s.
    for rate in (44100, 22050, 16000, 96000, 8000):
        fixture = workspace / f"rate_{rate}.wav"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                "-ac", "1", "-ar", str(rate), str(fixture),
            ],
            check=True,
        )
        clean = Step2Denoise(output_dir=workspace).run(
            DownloadedAudio(file=fixture, metadata=METADATA)
        )
        before = probe_wav(fixture)
        after = probe_wav(clean.file)
        drift = abs(after.duration - before.duration)
        assert drift <= MAX_DURATION_DRIFT, (
            f"{rate} Hz: {before.duration:.4f}s -> {after.duration:.4f}s "
            f"(drift {drift * 1000:.1f} ms)"
        )
        print(f"  {rate:>5} Hz: {before.duration:.4f}s -> {after.duration:.4f}s  ok")
    print("  ok")


def check_format(workspace: Path) -> None:
    print("== output format ==")
    fixture = build_fixture(workspace / "format.wav")
    clean = Step2Denoise(output_dir=workspace).run(
        DownloadedAudio(file=fixture, metadata=METADATA)
    )
    info = probe_wav(clean.file)
    print(f"  {info.sample_rate} Hz, {info.channels} ch, {info.sample_bits} bit")
    assert info.sample_bits == 16, info
    assert info.sample_rate == 48000, info
    assert clean.file.stem == "format_clean", clean.file
    print("  ok")


def check_metadata(workspace: Path) -> None:
    print("== metadata preserved ==")
    fixture = build_fixture(workspace / "metadata.wav")
    step = Step2Denoise(output_dir=workspace)
    clean = step.run(DownloadedAudio(file=fixture, metadata=METADATA))
    assert clean.metadata == METADATA, (clean.metadata, METADATA)
    print(f"  {clean.metadata.title} / {clean.metadata.artist}")
    print("  ok")


def check_heavy_noise(workspace: Path) -> None:
    print("== degrades gracefully under heavy noise ==")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anoisesrc=color=brown:amplitude=0.9:duration=3",
            "-f", "lavfi", "-i", flite_source(),
            "-filter_complex",
            "[0]aformat=sample_fmts=fltp:channel_layouts=mono[n];"
            "[1]aformat=sample_fmts=fltp:channel_layouts=mono[sp];"
            "[n][sp]amix=inputs=2:normalize=0,"
            "aresample=48000,aformat=channel_layouts=stereo",
            str(workspace / "heavy.wav"),
        ],
        check=True,
    )
    noisy = DownloadedAudio(file=workspace / "heavy.wav", metadata=METADATA)
    clean = Step2Denoise(output_dir=workspace).run(noisy)
    before, _ = read_samples(noisy.file)
    after, rate = read_samples(clean.file)
    print(f"  snr: {snr_db(before, 48000):.2f} dB -> {snr_db(after, rate):.2f} dB")
    assert not clean.is_empty()
    print("  ok")


def check_errors(workspace: Path) -> None:
    print("== errors ==")
    step = Step2Denoise(output_dir=workspace)

    try:
        step.run(DownloadedAudio(file=workspace / "nope.wav", metadata=METADATA))
    except StepOutputError as exc:
        assert "empty input" in str(exc), exc
    else:
        raise AssertionError("a missing input must raise StepOutputError")
    print("  missing input -> StepOutputError")

    # A corrupted model on disk must be detected, never handed to ffmpeg.
    model_dir = workspace / "arnndn"
    model_dir.mkdir()
    broken = model_dir / "std.rnnn"
    broken.write_bytes(b"not a real model")
    original_dir = step2_denoise.MODEL_DIR
    step2_denoise.MODEL_DIR = model_dir
    try:
        step = Step2Denoise(output_dir=workspace)
        assert not step._is_valid(broken, step2_denoise.MODELS["std"]["sha256"])
        print("  corrupted model on disk -> detected by checksum")

        # A download that returns the wrong bytes must be rejected outright.
        def fake_urlopen(url, timeout=None):
            return io.BytesIO(b"tampered payload")

        original_urlopen = step2_denoise.urllib.request.urlopen
        step2_denoise.urllib.request.urlopen = fake_urlopen
        try:
            step = Step2Denoise(output_dir=workspace)
            step.ensure_model()
        except StepOutputError as exc:
            assert "checksum" in str(exc), exc
            assert not (model_dir / "std.rnnn.download").exists(), "partial file left behind"
            print("  tampered download  -> StepOutputError")
        else:
            raise AssertionError("a model with the wrong checksum must never be used")
        finally:
            step2_denoise.urllib.request.urlopen = original_urlopen
    finally:
        step2_denoise.MODEL_DIR = original_dir


def check_model_idempotency() -> None:
    print("== model download is idempotent ==")
    step = Step2Denoise()
    first = step.ensure_model()
    stamp = first.stat().st_mtime_ns
    second = Step2Denoise().ensure_model()
    assert first == second, (first, second)
    assert second.stat().st_mtime_ns == stamp, "the model was re-downloaded"
    print(f"  {second} ({second.stat().st_size} bytes, not re-downloaded)")
    print("  ok")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        check_snr(workspace)
        check_duration(workspace)
        check_sample_rates(workspace)
        check_format(workspace)
        check_metadata(workspace)
        check_heavy_noise(workspace)
        check_errors(workspace)
    check_model_idempotency()
    print("\nall good")