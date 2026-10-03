#!/usr/bin/env python3
"""Smoke test for step 5.

Offline, deterministic and fast: the fixture is a synthesised sine sweep, since
this step only needs wav bytes and a duration, not speech. Step 3's smoke test
covers producing a real transcript and step 4's covers producing real timings.

The interesting check is ``sample accuracy``. The script slices the source with
the stdlib ``wave`` module — the byte-exact reference the PRD lists as the
alternative implementation — and compares the result byte for byte against what
FFmpeg produced. That turns "no temporal drift" from a tolerance into an exact
comparison.

The three success criteria of the PRD are covered by the first three groups.

    python scripts/smoke_step5.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import CleanedAudio, Metadata, Phrase, Phrases  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.media import probe_wav  # noqa: E402
from pipeline.steps.step5_cut import Step5Cut, _Cut  # noqa: E402

DURATION = 6.0
TITLE = "A Test Video: Part 1/2"

METADATA = Metadata(
    url="https://www.youtube.com/watch?v=fixture",
    source="youtube",
    title=TITLE,
    artist="Fixture Artist",
    album=None,
    source_id="fixture",
)

# The PRD's "~100 phrases per video".
SCALE_PHRASES = 100
MAX_SCALE_SECONDS = 15.0


def build_fixture(path: Path, duration: float = DURATION, rate: int = 48000) -> Path:
    """A deterministic multi-tone wav; content only matters for byte comparison."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"sine=frequency=220:sample_rate={rate}:duration={duration}",
            "-ac", "2", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path


def cleaned(path: Path) -> CleanedAudio:
    return CleanedAudio(file=path, metadata=METADATA)


def phrases(*spans: tuple[str, float, float]) -> Phrases:
    return Phrases(
        phrases=[
            Phrase(id=f"p{index:03d}", text=text, start=start, end=end)
            for index, (text, start, end) in enumerate(spans, start=1)
        ]
    )


def read(path: Path) -> tuple[bytes, wave.Wave_read]:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle


def reference_slice(source: Path, start_frame: int, end_frame: int) -> bytes:
    """Cut ``source`` with the stdlib, the byte-exact oracle for the test."""
    with wave.open(str(source), "rb") as handle:
        handle.setpos(start_frame)
        return handle.readframes(end_frame - start_frame)


def check_duration_and_margin(workspace: Path) -> None:
    print("== duration is phrase + 200 ms ==")
    fixture = build_fixture(workspace / "source.wav")
    info = probe_wav(fixture)

    # A phrase in the middle, one starting at 0 (margin clipped) and one ending
    # at the last sample (margin clipped). Margin can only shrink at the edges.
    wanted = phrases(
        ("in the middle of the audio", 2.0, 3.0),
        ("at the very beginning", 0.05, 0.5),
        ("at the very end", 5.6, 6.0),
    )
    result = Step5Cut(output_dir=workspace).run(cleaned(fixture), wanted)

    assert len(result.fragments) == 3, result.fragments
    for fragment, phrase in zip(result.fragments, wanted.phrases):
        text, start, end = phrase.text, phrase.start, phrase.end
        margin_start = max(0.0, start - 0.1)
        margin_end = min(DURATION, end + 0.1)
        expected = round(margin_end * info.sample_rate) - round(
            margin_start * info.sample_rate
        )
        got = probe_wav(fragment.file).frames
        assert got == expected, (fragment, expected)
        print(f"  {fragment.file.name} {got} samples "
              f"({fragment.start:.3f}s..{fragment.end:.3f}s)")

    # The middle one is the case the PRD describes literally.
    middle = probe_wav(result.fragments[0].file).duration
    assert abs(middle - 1.2) < 1e-9, middle
    # The two edge ones are clipped, never extended past the file.
    assert probe_wav(result.fragments[1].file).duration < 1.2 - 0.1
    assert probe_wav(result.fragments[2].file).duration < 1.2 - 0.1
    print("  ok")


def check_sample_accuracy(workspace: Path) -> None:
    print("== cuts are sample accurate ==")
    fixture = build_fixture(workspace / "accuracy.wav")
    info = probe_wav(fixture)

    # Times that do not land on a sample boundary: 1/48000 s apart neighbours
    # would round to different frames, so this proves the rounding is explicit.
    spans = [("first fragment here", 0.2537, 0.7537), ("second fragment here", 2.99997, 4.00003)]
    result = Step5Cut(output_dir=workspace).run(cleaned(fixture), phrases(*spans))

    source, _ = read(fixture)
    for fragment, (text, start, end) in zip(result.fragments, spans):
        del text  # only the span matters below
        first = round((start - 0.1) * info.sample_rate)
        last = round((end + 0.1) * info.sample_rate)
        expected = source[first * 4 : last * 4]
        got, _ = read(fragment.file)
        assert got == expected, f"{fragment.file.name} differs from the source bytes"
        # And the reported timings address exactly those samples.
        assert round(fragment.start * info.sample_rate) == first, fragment
        assert round(fragment.end * info.sample_rate) == last, fragment
        print(f"  {fragment.file.name} byte-identical to samples [{first}, {last})")
    print("  ok")


def check_lossless_and_times(workspace: Path) -> None:
    print("== fragments stay lossless and carry their timings ==")
    fixture = build_fixture(workspace / "format.wav")
    source = probe_wav(fixture)
    result = Step5Cut(output_dir=workspace).run(
        cleaned(fixture), phrases(("one fragment", 1.0, 2.0))
    )

    fragment = result.fragments[0]
    info = probe_wav(fragment.file)
    assert info.sample_rate == source.sample_rate, (info, source)
    assert info.channels == source.channels, (info, source)
    assert info.sample_bits == source.sample_bits, (info, source)
    # pcm_s16le, not a re-encode: the bytes match the source exactly.
    expected = reference_slice(
        fixture,
        round(fragment.start * source.sample_rate),
        round(fragment.end * source.sample_rate),
    )
    assert read(fragment.file)[0] == expected, fragment
    assert fragment.id == "p001", fragment
    assert fragment.end > fragment.start, fragment
    print(f"  {info.sample_rate} Hz, {info.channels} ch, {info.sample_bits} bit, "
          f"id={fragment.id}")
    print("  ok")


def check_file_layout(workspace: Path) -> None:
    print("== one readable file per phrase ==")
    fixture = build_fixture(workspace / "layout.wav")
    # Duplicated text, an awkward title and punctuation: the names must stay
    # unique and the folder must still be identified by the video.
    wanted = phrases(
        ("I am twenty years old", 0.5, 1.0),
        ("I am twenty years old", 1.5, 2.0),
        ("Let's go: \"now\"! (really?)", 2.5, 3.0),
    )
    step = Step5Cut(output_dir=workspace)
    result = step.run(cleaned(fixture), wanted)

    files = [fragment.file for fragment in result.fragments]
    assert len({path.name for path in files}) == 3, files
    assert all(path.exists() and path.stat().st_size > 44 for path in files), files
    assert all(path.parent == files[0].parent for path in files), files
    # Per-video folder, named after the metadata title.
    assert files[0].parent.name == "a_test_video_part_1_2", files[0].parent
    assert files[0].parent.parent.name == "fragments", files[0].parent
    for fragment in result.fragments:
        print(f"  {fragment.file.relative_to(workspace)}")
    print("  ok")


def check_margin_zero(workspace: Path) -> None:
    print("== margin=0 cuts the phrase exactly ==")
    fixture = build_fixture(workspace / "nomargin.wav")
    info = probe_wav(fixture)
    result = Step5Cut(output_dir=workspace, margin=0.0).run(
        cleaned(fixture), phrases(("exactly this", 1.0, 2.0))
    )
    fragment = result.fragments[0]
    assert probe_wav(fragment.file).frames == info.sample_rate, fragment
    assert (fragment.start, fragment.end) == (1.0, 2.0), fragment
    print(f"  {probe_wav(fragment.file).frames} samples, {fragment.start}..{fragment.end}")
    print("  ok")


def check_unusable_phrases(workspace: Path) -> None:
    print("== out of range phrases are reported, not silently dropped ==")
    fixture = build_fixture(workspace / "range.wav")

    step = Step5Cut(output_dir=workspace)
    result = step.run(
        cleaned(fixture),
        phrases(
            ("perfectly fine", 1.0, 2.0),
            ("long after the audio ends", 30.0, 31.0),
            ("an inverted span", 4.0, 3.0),
        ),
    )
    assert len(result.fragments) == 1, result.fragments
    # The numbering keeps the position in Phrases, so a skip leaves a gap
    # instead of renaming every later fragment.
    assert result.fragments[0].file.name.startswith("001_"), result.fragments[0].file
    assert [miss.index for miss in step.skipped] == [2, 3], step.skipped
    # Each skip says what is actually wrong: past the end of the file versus a
    # span that runs backwards.
    assert "falls outside" in step.skipped[0].reason, step.skipped[0]
    assert "ends before it starts" in step.skipped[1].reason, step.skipped[1]
    print(step.report().splitlines()[0])
    for line in step.report().splitlines()[1:]:
        print(f"  {line.strip()}")

    # A zero-length phrase is still cuttable: the margin alone gives it the
    # 200 ms of audio around the point it refers to.
    step = Step5Cut(output_dir=workspace)
    result = step.run(cleaned(fixture), phrases(("a zero length phrase", 3.0, 3.0)))
    assert not step.skipped, step.report()
    assert abs(probe_wav(result.fragments[0].file).duration - 0.2) < 1e-9
    print("  zero length span -> still cut, 0.200 s of margin around it")

    # A phrase that reaches past the end is clamped, not refused: the audio
    # still contains most of it.
    step = Step5Cut(output_dir=workspace)
    result = step.run(cleaned(fixture), phrases(("runs off the end", 5.8, 12.0)))
    assert not step.skipped, step.report()
    assert abs(result.fragments[0].end - DURATION) < 1e-9, result.fragments[0]
    print(f"  clamped to the end of the file at {result.fragments[0].end:.3f}s")
    print("  ok")


def check_scale(workspace: Path) -> None:
    print("== scale ==")
    fixture = build_fixture(workspace / "scale.wav", duration=120.0)
    wanted = Phrases(
        phrases=[
            Phrase(id=f"p{index:03d}", text=f"phrase number {index}", start=1.0 + index * 0.9, end=1.6 + index * 0.9)
            for index in range(SCALE_PHRASES)
        ]
    )
    step = Step5Cut(output_dir=workspace)
    started = time.monotonic()
    result = step.run(cleaned(fixture), wanted)
    elapsed = time.monotonic() - started

    assert len(result.fragments) == SCALE_PHRASES, len(result.fragments)
    assert not step.skipped, step.report()
    print(f"  {SCALE_PHRASES} fragments from a 2 minute wav in {elapsed:.2f}s")
    assert elapsed < MAX_SCALE_SECONDS, f"took {elapsed:.1f}s, cutting is too slow"

    # batch_size=1 must produce byte-identical output: batching is a transport
    # detail, never a change in the result.
    step = Step5Cut(output_dir=workspace, batch_size=7)
    batched = step.run(cleaned(fixture), wanted)
    for a, b in zip(result.fragments, batched.fragments):
        assert a.file.parent.name == b.file.parent.name, (a, b)
        assert (a.start, a.end) == (b.start, b.end), (a, b)
        assert read(a.file)[0] == read(b.file)[0], a
    print("  batch_size=7 -> identical fragments")
    print("  ok")


def check_errors(workspace: Path) -> None:
    print("== errors ==")
    fixture = build_fixture(workspace / "errors.wav")
    step = Step5Cut(output_dir=workspace)

    try:
        step.run(cleaned(workspace / "nope.wav"), phrases(("hello", 1.0, 2.0)))
    except StepOutputError as exc:
        assert "empty input" in str(exc), exc
    else:
        raise AssertionError("a missing input must raise StepOutputError")
    print("  missing input -> StepOutputError")

    try:
        step.run(cleaned(fixture), Phrases())
    except StepOutputError as exc:
        assert "no phrases" in str(exc), exc
    else:
        raise AssertionError("an empty Phrases must raise StepOutputError")
    print("  no phrases -> StepOutputError")

    try:
        step.run(cleaned(fixture), phrases(("entirely past the end", 40.0, 41.0)))
    except StepOutputError as exc:
        assert "could not cut any" in str(exc), exc
    else:
        raise AssertionError("no cuttable phrase must raise StepOutputError")
    print("  nothing cuttable -> StepOutputError")

    for kwargs in ({"margin": -1.0}, {"batch_size": 0}):
        try:
            Step5Cut(**kwargs)
        except StepOutputError:
            print(f"  bad {kwargs} -> StepOutputError")
        else:
            raise AssertionError(f"Step5Cut({kwargs}) should have been rejected")

    # A fragment that does not match its reported timings must be caught after
    # the fact, not trusted because ffmpeg exited 0.
    good = step.run(cleaned(fixture), phrases(("something", 1.0, 2.0)))
    target = good.fragments[0]
    rate = probe_wav(fixture).sample_rate
    honest = _Cut(
        1,
        target.id,
        "something",
        target.file,
        round(target.start * rate),
        round(target.end * rate),
        rate,
    )
    Step5Cut._verify(honest)
    tampered = _Cut(
        1, target.id, "something", target.file, honest.start_frame, honest.end_frame // 2, rate
    )
    try:
        Step5Cut._verify(tampered)
    except StepOutputError as exc:
        assert "were cut" in str(exc), exc
    else:
        raise AssertionError("a short fragment must be detected")
    print("  wrong sample count -> StepOutputError")
    print("  ok")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        check_duration_and_margin(workspace)
        check_sample_accuracy(workspace)
        check_lossless_and_times(workspace)
        check_file_layout(workspace)
        check_margin_zero(workspace)
        check_unusable_phrases(workspace)
        check_scale(workspace)
        check_errors(workspace)
    print("\nall good")