#!/usr/bin/env python3
"""Smoke test for step 4.

Offline, deterministic and fast: the transcript is built by hand instead of
running Whisper, because this step only needs words with plausible timings and
must be testable in a fraction of a second. Step 3's own smoke test covers
producing a real transcript.

The three success criteria of the PRD map onto the first three checks: timings
inside the word boundaries, the first occurrence winning, and phrases that are
not in the transcript being reported.

    python scripts/smoke_step4.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import Transcript, Word  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.steps.step4_align import Step4Align  # noqa: E402

WORD_DURATION = 0.4
LINE_GAP = 0.2

# 5 000 words is a two-hour conversation, and 2 000 phrases is far past the ~100
# the PRD expects; the run must still feel instant.
SCALE_WORDS = 5000
SCALE_PHRASES = 2000
MAX_SCALE_SECONDS = 10.0


def make_transcript(lines: list[str]) -> Transcript:
    """Build a ``Transcript`` where each word gets a predictable slot."""
    words: list[Word] = []
    clock = 0.0
    for line in lines:
        for raw in line.split():
            words.append(Word(word=raw, start=round(clock, 3), end=round(clock + WORD_DURATION, 3)))
            clock += WORD_DURATION
        clock += LINE_GAP
    text = " ".join(" ".join(line.split()) for line in lines)
    return Transcript(text=text, language="en", words=words)


def slot_of(transcript: Transcript, word: str, occurrence: int = 0) -> Word:
    """The word of ``transcript`` at a given repetition, for exact expectations."""
    seen = [w for w in transcript.words if w.word == word]
    return seen[occurrence]


def check_exact_alignment() -> None:
    print("== exact alignment ==")
    transcript = make_transcript(
        [
            "the quick brown fox jumps over the lazy dog",
            "she is going to the store right now",
        ]
    )
    step = Step4Align()
    phrases = step.run(transcript, ["jumps over the lazy dog", "going to the store"])
    assert len(phrases.phrases) == 2, phrases.phrases
    assert not step.missing, step.report()

    for phrase, wanted in zip(phrases.phrases, ["jumps over the lazy dog", "going to the store"]):
        first, last = wanted.split()[0], wanted.split()[-1]
        # Criterion 1: the span has to sit inside the words it matched.
        assert phrase.start == slot_of(transcript, first).start, phrase
        assert phrase.end == slot_of(transcript, last).end, phrase
        assert phrase.end > phrase.start, phrase
        print(f"  {phrase.id} {phrase.text!r} -> {phrase.start:.2f}s..{phrase.end:.2f}s")

    # The text is preserved verbatim, punctuation and all.
    assert phrases.phrases[0].text == "jumps over the lazy dog"


def check_occurrences() -> None:
    print("== first occurrence wins ==")
    transcript = make_transcript(["we should leave now", "we should leave now"])
    wanted = "we should leave now"

    first = Step4Align().run(transcript, [wanted])
    assert first.phrases[0].start == slot_of(transcript, "we", 0).start, first.phrases
    print(f"  first_only -> {first.phrases[0].start:.2f}s..{first.phrases[0].end:.2f}s")

    # Criterion 2, spelled out: the second occurrence starts later.
    every = Step4Align(first_only=False).run(transcript, [wanted])
    assert len(every.phrases) == 2, every.phrases
    starts = [p.start for p in every.phrases]
    assert starts == sorted(starts), starts
    assert starts[0] < starts[1], starts
    assert starts[1] == slot_of(transcript, "we", 1).start, every.phrases
    assert len({p.id for p in every.phrases}) == 2, "ids must stay unique"
    print(f"  all        -> {[p.id for p in every.phrases]} at {starts}")


def check_normalization() -> None:
    print("== contractions, case and punctuation ==")
    transcript = make_transcript(
        ["I'm twenty years old and I don't like coffee", "let's go to the store"]
    )
    # Same wording, written the other way round: contractions are expanded on
    # both sides, so both spellings must land on the same span.
    # (phrase as the user writes it, first word, last word) of what is spoken.
    pairs = [
        ("I am twenty years old", "I'm", "old"),
        ("I'm twenty years old", "I'm", "old"),
        ("I do not like coffee", "I", "coffee"),
        ("LET'S GO TO THE STORE!", "let's", "store"),
        ("let us go to the store", "let's", "store"),
        ("Let’s go to the store.", "let's", "store"),
        ("Twenty Years Old.", "twenty", "old"),
    ]
    step = Step4Align()
    for phrase, first_word, last_word in pairs:
        found = step.run(transcript, [phrase]).phrases[0]
        assert found.start == slot_of(transcript, first_word).start, (phrase, found)
        assert found.end == slot_of(transcript, last_word).end, (phrase, found)
        assert not step.alignments[0].fuzzy, f"{phrase!r} should match exactly"
    print(f"  {len(pairs)} spellings of the same words all matched exactly")


def check_fuzzy() -> None:
    print("== fuzzy recovery ==")
    transcript = make_transcript(["the quick brown fox jumps over the lazy dog"])

    # A misheard word: no exact run, but the window is recognisably the same.
    step = Step4Align()
    found = step.run(transcript, ["the quick brown foxes"]).phrases[0]
    assert step.alignments[0].fuzzy, "a misheard word should be recovered fuzzily"
    assert found.start == slot_of(transcript, "the").start, found
    assert found.end == slot_of(transcript, "fox").end, found
    print(f"  'the quick brown foxes' -> {found.start:.2f}s..{found.end:.2f}s "
          f"({step.alignments[0].score:.0f}% fuzzy)")

    # Off by more than the threshold: reported, not guessed. The first phrase
    # shares its first word with the transcript but scores too low; the second
    # shares no word at all, so there is no window left to score. With nothing
    # matched the step stops the flow, but the report is still available.
    step = Step4Align()
    try:
        step.run(
            transcript,
            ["the quick summer weather today", "zeppelin marmalade"],
        )
    except StepOutputError as exc:
        assert "could not place any" in str(exc), exc
    else:
        raise AssertionError("two unrelated phrases must not produce any alignment")
    assert len(step.alignments) == 0, step.report()
    assert len(step.missing) == 2, step.report()
    low, absent = step.missing
    assert "below" in low.reason, low
    assert low.best_score is not None and low.best_score < 85.0, low
    assert "none of its words" in absent.reason, absent
    assert absent.best_score is None, absent
    print(f"  unrelated phrase -> reported ({low.reason})")
    print(f"  absent words     -> reported ({absent.reason})")

    # Criterion 3: a phrase that is not there is reported, and the others still
    # come back aligned.
    step = Step4Align()
    phrases = step.run(transcript, ["lazy dog", "nothing like this is spoken here"])
    assert len(phrases.phrases) == 1, phrases
    assert step.missing[0].text == "nothing like this is spoken here", step.missing
    print(step.report().splitlines()[0])

    # allow_fuzzy=False keeps the dependency out and reports exact misses only.
    step = Step4Align(allow_fuzzy=False)
    step.run(transcript, ["lazy dog", "the quick brown foxes"])
    assert step.missing[0].reason == "no occurrence in the transcript", step.missing[0]
    print("  allow_fuzzy=False -> plain 'no occurrence'")


def check_scale() -> None:
    print("== scale ==")
    vocabulary = [f"w{index}" for index in range(50)]
    line = " ".join(vocabulary[index % 50] for index in range(SCALE_WORDS))
    transcript = make_transcript([line])
    assert len(transcript.words) == SCALE_WORDS

    phrases = [
        " ".join(word.word for word in transcript.words[start : start + 4])
        for start in range(0, SCALE_PHRASES * 2, 2)
    ]
    step = Step4Align()
    started = time.monotonic()
    result = step.run(transcript, phrases)
    elapsed = time.monotonic() - started

    assert len(result.phrases) == len(phrases), len(result.phrases)
    assert not step.missing, step.report()
    print(f"  {SCALE_WORDS} words x {len(phrases)} phrases in {elapsed:.2f}s")
    assert elapsed < MAX_SCALE_SECONDS, f"took {elapsed:.1f}s, alignment is too slow"


def check_errors() -> None:
    print("== errors ==")
    transcript = make_transcript(["the quick brown fox"])
    step = Step4Align()

    try:
        step.run(Transcript(text="", language="en", words=[]), ["the quick"])
    except StepOutputError as exc:
        assert "empty transcript" in str(exc), exc
    print("  empty transcript -> StepOutputError")

    try:
        step.run(transcript, [])
    except StepOutputError as exc:
        assert "no phrases" in str(exc), exc
    print("  no phrases -> StepOutputError")

    try:
        step.run(transcript, ["utterly absent from this transcript"])
    except StepOutputError as exc:
        assert "could not place any" in str(exc), exc
    print("  nothing matched -> StepOutputError")

    for kwargs in ({"fuzzy_threshold": 120.0}, {"window_slack": -1}):
        try:
            Step4Align(**kwargs)
        except StepOutputError as exc:
            print(f"  bad {kwargs} -> StepOutputError")
        else:
            raise AssertionError(f"Step4Align({kwargs}) should have been rejected")


def check_ids() -> None:
    print("== ids ==")
    transcript = make_transcript(["one two three", "four five six", "seven eight nine"])
    step = Step4Align()
    phrases = step.run(transcript, ["four five six", "absent phrase here", "one two three"])
    ids = [p.id for p in phrases.phrases]
    assert ids == ["p001", "p002"], ids
    assert len(set(ids)) == len(ids), ids
    # Ids index the matched phrases, so a miss does not leave a hole.
    print(f"  {ids} (a missed phrase does not consume an id)")


if __name__ == "__main__":
    check_exact_alignment()
    check_occurrences()
    check_normalization()
    check_fuzzy()
    check_ids()
    check_scale()
    check_errors()
    print("\nall good")