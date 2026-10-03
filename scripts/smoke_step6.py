#!/usr/bin/env python3
"""Smoke test for step 6.

Two layers, because step 6 is the only step whose engine is a language model:

* **Offline** — the parsing, cleanup and guard logic, which is where a silent
  bug would put the wrong Spanish on a card. These run with no server and no
  network, and every one of them corresponds to a failure actually measured
  against a real model.
* **Live** — a real translation through ``llama-server``. Skipped, loudly, when
  no server answers, so the offline layer stays usable on a bare checkout.

The offline layer is not decoration. Each case below is a reply that a 1-1.5B
model really produced during development: numerals rewritten as Spanish words,
an ignored JSON schema, an example parroted back as an answer, a subject
hedged as "he/she/it", and the PRD's own reference translation being thrown away
by an over-eager guard.

    python scripts/smoke_step6.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import (  # noqa: E402
    CleanedAudio,
    Fragment,
    Fragments,
    Metadata,
    Phrase,
    Phrases,
)
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.steps.step5_cut import Step5Cut  # noqa: E402
from pipeline.steps.step6_translate import (  # noqa: E402
    FEW_SHOT,
    Step6Translate,
    canonical_source,
)

METADATA = Metadata(
    url="https://www.youtube.com/watch?v=fixture",
    source="youtube",
    title="Fixture Video",
    artist="Fixture Artist",
    album=None,
    source_id="fixture",
)

BASE_URL = "http://127.0.0.1:8080"

# The PRD's acceptance case, and the ones a small model actually gets wrong.
ACCEPTANCE = [
    ("I'm twenty years old", "Tengo 20 años"),
    ("She is going to the store", "Está saliendo"),
    ("You look great today", "Te ves genial"),
]


def server_is_up(base_url: str = BASE_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- offline


def check_tag_parsing() -> None:
    print("== replies are parsed by tag, not by position ==")
    parse = Step6Translate._parse

    # Measured: asked for numbered lines, the model answered in Spanish words.
    spanish_numbers = "Uno: Tengo 20 años\nDos: Vámonos\nTres: No será fácil"
    assert parse(spanish_numbers) == {}, parse(spanish_numbers)
    print("  'Uno:/Dos:/Tres:' -> no tags, nothing trusted")

    # Measured: asked for a JSON object, it ignored the schema entirely.
    ignored_schema = "".join('{"n": 1, "v": "es"}\n' for _ in range(14))
    assert parse(ignored_schema) == {}, parse(ignored_schema)
    print("  schema-less JSON -> no tags, nothing trusted")

    good = "[[1]] Tengo 20 años\n[[2]] No será fácil\n[[3]] Vámonos"
    assert parse(good) == {"1": "Tengo 20 años", "2": "No será fácil", "3": "Vámonos"}
    # A preamble before the first tag is dropped, not parsed as a translation.
    assert parse("Aquí te dejo la traducción:\n" + good)["1"] == "Tengo 20 años"
    # A repeated tag collapses to one entry, so it cannot be counted twice.
    assert parse("[[1]] a\n[[1]] b") == {"1": "b"}
    print("  tagged lines -> parsed, preamble dropped, repeats collapsed")


def check_cleanup() -> None:
    print("== line cleanup ==")
    clean = Step6Translate._clean
    assert clean("  Tengo 20 años.  ") == "Tengo 20 años"
    # Measured: the model sometimes answers across two lines.
    assert clean("¿Qué estás haciendo esta noche\nSalir") == "¿Qué estás haciendo esta noche"
    assert clean("Vámonos — es tarde") == "Vámonos"
    assert clean("No será fácil - de verdad") == "No será fácil"
    print("  padding, trailing punctuation and extra lines dropped")


def check_leak_guard() -> None:
    print("== parroting an example is rejected ==")
    step = Step6Translate()

    # Measured: the model answered with example 2's translation.
    assert step._is_leak("She is going to the store", "Vámonos que llegamos tarde")
    print("  verbatim example -> rejected")

    # But matching the example is correct when the phrase *is* the example,
    # including when it is spelled differently ("20" vs "twenty"). Getting this
    # wrong threw away the PRD's own reference translation.
    assert not step._is_leak("I'm twenty years old", "Tengo 20 años")
    assert not step._is_leak("I'm 20 years old", "Tengo 20 años")
    print("  the example's own phrase -> accepted (both spellings)")

    # An unrelated phrase is not excused by any example.
    assert not step._is_leak("She is going to the store", "Está saliendo al supermercado")
    print("  unrelated phrase -> accepted")


def check_number_canonicalisation() -> None:
    print("== spelled numbers are the same phrase ==")
    assert canonical_source("I'm twenty years old") == canonical_source("I'm 20 years old")
    assert canonical_source("I have forty five dollars") == "i have 45 dollars"
    assert Step6Translate._same_phrase("I've been here", "I have been here")
    assert not Step6Translate._same_phrase("Let's get going", "We should leave now")
    print("  'twenty' == '20', contractions folded, different phrases kept apart")


def check_untranslated_guard() -> None:
    print("== replies that are not translations are rejected ==")
    step = Step6Translate()
    # Measured: the model hedged the subject instead of choosing one.
    hedged = "He/she/it has been learning English for two years"
    assert step._looks_untranslated("I have been learning English for two years", hedged)
    # Measured: the English came straight back.
    assert step._looks_untranslated("The book is on the table", "The book is on the table")
    # Real translations must survive.
    for english, spanish in ACCEPTANCE:
        assert not step._looks_untranslated(english, spanish), (english, spanish)
    print("  'he/she/it' and English echoes rejected, translations kept")


def check_prompt_shape() -> None:
    print("== the prompt cannot leak a tag ==")
    step = Step6Translate()
    fragment = Fragment(id="p001", file=Path("x.wav"), start=0.0, end=1.0)
    from pipeline.steps.step6_translate import _Pending

    chunk = [_Pending(1, "p001", "She is going to the store", fragment)]
    messages = step._messages(chunk)

    assert messages[0]["role"] == "system"
    # The examples are real turns, so their [[1]] tags live in earlier messages
    # and never collide with the request. Flattening them into one user message
    # is what let the model answer with example 2's translation.
    assert sum(1 for m in messages if m["role"] == "assistant") == len(FEW_SHOT)
    final = messages[-1]
    assert final["role"] == "user" and "[[1]] She is going to the store" in final["content"]
    assert "[[1]]" not in "\n".join(m["content"] for m in messages[:-1]).split("[[1]] ")[0]
    print(f"  {len(messages)} messages, examples in {len(FEW_SHOT)} assistant turns")


def check_errors() -> None:
    print("== errors ==")
    fragments = Fragments(
        fragments=[Fragment(id="p001", file=Path("/nope/missing.wav"), start=0.0, end=1.0)]
    )
    good = Phrases(phrases=[Phrase(id="p001", text="hello", start=0.0, end=1.0)])

    step = Step6Translate(base_url="http://127.0.0.1:9")
    try:
        step.require_server()
    except StepOutputError as exc:
        assert "llama-server" in str(exc) and "Start it with" in str(exc), exc
    else:
        raise AssertionError("an unreachable server must raise StepOutputError")
    print("  no server -> StepOutputError with instructions")

    step = Step6Translate()
    for label, phrases_in, fragments_in, expected in (
        ("empty phrases", Phrases(), fragments, "no phrases"),
        ("empty fragments", good, Fragments(), "no fragments"),
    ):
        try:
            step.run(phrases_in, fragments_in)
        except StepOutputError as exc:
            assert expected in str(exc), (label, exc)
        else:
            raise AssertionError(f"{label} must raise StepOutputError")
    print("  empty input -> StepOutputError")

    try:
        Step6Translate(batch_phrases=0)
    except StepOutputError:
        print("  batch_phrases=0 -> StepOutputError")
    else:
        raise AssertionError("batch_phrases=0 must be rejected")


# ------------------------------------------------------------------ live


def make_workspace() -> tuple[Path, Phrases, Fragments]:
    """A real audio file cut into real fragments, for the live layer."""
    workspace = Path(tempfile.mkdtemp())
    source = workspace / "source.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "sine=frequency=220:sample_rate=48000:duration=30",
            "-ac", "2", "-c:a", "pcm_s16le", str(source),
        ],
        check=True,
    )
    spans = [(english, 1.0 + i * 3.0, 2.0 + i * 3.0) for i, (english, _) in enumerate(ACCEPTANCE)]
    phrases = Phrases(
        phrases=[
            Phrase(id=f"p{index:03d}", text=text, start=start, end=end)
            for index, (text, start, end) in enumerate(spans, start=1)
        ]
    )
    fragments = Step5Cut(output_dir=workspace).run(
        CleanedAudio(file=source, metadata=METADATA), phrases
    )
    return workspace, phrases, fragments


def check_live_translation() -> None:
    print("== live translation ==")
    workspace, phrases, fragments = make_workspace()
    step = Step6Translate()
    started = time.monotonic()
    dataset = step.run(phrases, fragments)
    elapsed = time.monotonic() - started

    assert len(dataset.cards) == len(ACCEPTANCE), (step.report(), dataset.cards)
    print(f"  {len(dataset.cards)}/{len(phrases.phrases)} phrases in {elapsed:.1f}s")

    by_id = {card.id: card for card in dataset.cards}
    # Criterion 1: idiomatic, never a literal calque.
    assert by_id["p001"].es == "Tengo 20 años", by_id["p001"].es
    print(f"  p001 {by_id['p001'].en!r} -> {by_id['p001'].es!r}")

    # The card joins the English text, the audio and the timings.
    for card, fragment in zip(dataset.cards, fragments.fragments):
        assert card.audio == fragment.file, card
        assert (card.start, card.end) == (fragment.start, fragment.end), card
        with wave.open(str(card.audio), "rb") as handle:
            assert handle.getnframes() > 0
    assert all(card.en and card.es for card in dataset.cards)
    assert dataset.cards[0].id == "p001"
    print("  every card carries en + es + an existing wav + the fragment timings")
    print(f"  stats={step.stats}")


def check_live_consumer_agnostic() -> None:
    print("== the dataset knows nothing about the consumer ==")
    workspace, phrases, fragments = make_workspace()
    dataset = Step6Translate().run(phrases, fragments)
    field_names = {f.name for f in dataset.cards[0].__dataclass_fields__.values()}
    assert field_names == {"id", "en", "es", "audio", "start", "end"}, field_names
    # No Anki, no deck, no apkg anywhere in what step 6 produced.
    blob = repr(dataset).lower()
    for forbidden in ("anki", "deck", "apkg", "genanki"):
        assert forbidden not in blob, forbidden
    print(f"  card fields = {sorted(field_names)}, no consumer in sight")


if __name__ == "__main__":
    check_tag_parsing()
    check_cleanup()
    check_leak_guard()
    check_number_canonicalisation()
    check_untranslated_guard()
    check_prompt_shape()
    check_errors()

    if server_is_up():
        check_live_translation()
        check_live_consumer_agnostic()
    else:
        print("\n!! no llama-server on 127.0.0.1:8080: the live layer was skipped")
        print("!! start it with:")
        print(f"!!   llama-server -m {Step6Translate().model} --host 127.0.0.1 --port 8080 -c 4096 --threads 4")
        sys.exit(1)

    print("\nall good")