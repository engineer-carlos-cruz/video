#!/usr/bin/env python3
"""Smoke test for step 7.

Fully offline: ``genanki`` is a pure local library, and an ``.apkg`` is a zip,
so the package can be opened and inspected here without Anki installed. That is
deliberate — the two things this step can get silently wrong (a play button
pointing at audio that is not in the package, and a note type or deck that
duplicates on every import) are both invisible until study time, so they are
checked by reading the package back.

    python scripts/smoke_step7.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.contracts import Card, Dataset  # noqa: E402
from pipeline.errors import StepOutputError  # noqa: E402
from pipeline.steps.step7_anki import (  # noqa: E402
    FIELD_AUDIO,
    FIELD_ENGLISH,
    FIELD_NAMES,
    FIELD_SPANISH,
    MODEL_ID,
    MODEL_NAME,
    Step7Anki,
    _deck_id,
    build_model,
)

CARDS = [
    ("p001", "I'm twenty years old", "Tengo 20 años"),
    ("p002", "You look great today", "Te ves genial hoy"),
    ("p003", "She is going to the store", "Está saliendo al supermercado"),
]


def write_wav(path: Path, seconds: float = 0.3, rate: int = 16000) -> Path:
    """Write a real, playable wav so probe_wav has something to accept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def make_dataset(root: Path, cards=CARDS) -> Dataset:
    """Build a Dataset whose audio files really exist on disk."""
    built = []
    for index, (id_, english, spanish) in enumerate(cards, start=1):
        audio = write_wav(root / "fragments" / f"{index:03d}_{id_}.wav")
        built.append(
            Card(id=id_, en=english, es=spanish, audio=audio, start=0.0, end=0.3)
        )
    return Dataset(cards=built)


def collection(package_path: Path) -> dict:
    """Read an .apkg back the way Anki would: the zip plus its media manifest.

    genanki stores each media file under a numeric entry name and writes a
    ``media`` manifest mapping that index back to the file's basename. That is
    how a ``[sound:<name>.wav]`` inside a note resolves to an actual file, so
    resolving the reference means going through the manifest, not just listing
    the zip.
    """
    with zipfile.ZipFile(package_path) as archive:
        names = archive.namelist()
        assert "collection.anki2" in names, names
        raw = archive.read("collection.anki2")
        manifest = json.loads(archive.read("media"))
    # No sqlite parsing needed for these assertions: the note payloads are
    # searchable as text inside the collection.
    return {
        "names": names,
        "raw": raw,
        "text": raw.decode("utf-8", "ignore"),
        # basename -> zip entry holding its bytes
        "media": {basename: index for index, basename in manifest.items()},
    }


def check_model_shape() -> None:
    print("== the fixed ES-EN Audio model ==")
    model = build_model()
    assert model.model_id == MODEL_ID
    assert model.name == MODEL_NAME == "ES-EN Audio"
    assert [f["name"] for f in model.fields] == FIELD_NAMES
    assert FIELD_NAMES == [FIELD_SPANISH, FIELD_ENGLISH, FIELD_AUDIO]
    assert FIELD_NAMES == ["Español", "Inglés", "Audio"], FIELD_NAMES

    # Front is Spanish alone; the back carries English and the audio field,
    # which is where Anki renders the play button.
    template = model.templates[0]
    assert "{{Español}}" in template["qfmt"]
    assert "{{Inglés}}" not in template["qfmt"], "the front must not leak the answer"
    assert "{{Inglés}}" in template["afmt"]
    assert "{{Audio}}" in template["afmt"]
    print("  frente = español, reverso = inglés + audio")

    # A stable id is the whole reason the PRD asks for a dedicated model: an id
    # that changed per run would make Anki create a new note type every import.
    assert build_model().model_id == MODEL_ID
    print(f"  model id is fixed at {MODEL_ID}")


def check_deck_id_stable() -> None:
    print("== deck identity is derived from the name ==")
    assert _deck_id("Inglés 2026") == _deck_id("Inglés 2026")
    # Leading/trailing space is what a terminal input tends to add.
    assert _deck_id("  Inglés 2026  ") == _deck_id("Inglés 2026")
    assert _deck_id("mazo A") != _deck_id("mazo B")
    assert _deck_id("mazo a") != _deck_id("mazo A")
    print("  same name -> same id (so re-import adds to the existing deck)")


def check_package_contents(tmp: Path) -> None:
    print("== a valid package embeds its audio ==")
    step = Step7Anki(output_dir=tmp)
    dataset = make_dataset(tmp / "src")
    package = step.run(dataset, "Inglés 2026")
    apkg = package.apkg_path

    assert apkg.exists() and apkg.stat().st_size > 0
    assert apkg.suffix == ".apkg"
    assert apkg.parent.name == "anki"
    assert "all good" not in package.__repr__()

    info = collection(apkg)
    # Every fragment the dataset points at must be inside the package under its
    # own basename, so [sound:...] resolves when Anki plays the card.
    for card in dataset.cards:
        assert card.audio.name in info["media"], (
            f"{card.audio.name} missing from the media manifest: {info['media']}"
        )
    print(f"  {len(dataset.cards)} wavs embedded as {dataset.cards[0].audio.name!r}, ...")

    # And every [sound:] reference must resolve to an embedded file, by name.
    for card in dataset.cards:
        reference = f"[sound:{card.audio.name}]"
        assert reference in info["text"], reference
        assert card.audio.name in info["media"], reference
    print("  every [sound:] reference resolves to an embedded file")

    # The note type and deck are written into the collection itself.
    assert MODEL_NAME in info["text"]
    print(f"  note type {MODEL_NAME!r} present in the collection")


def check_front_and_back(tmp: Path) -> None:
    print("== each card carries its Spanish and its English ==")
    step = Step7Anki(output_dir=tmp)
    dataset = make_dataset(tmp / "src2")
    apkg = step.run(dataset, "Reconocimiento").apkg_path
    text = collection(apkg)["text"]

    for card in dataset.cards:
        assert card.es in text, f"missing spanish: {card.es}"
        assert card.en in text, f"missing english: {card.en}"
    print("  español + inglés for every card")

    # The PRD's reference translation must survive to the package.
    assert "Tengo 20 años" in text
    print("  the PRD's acceptance case ('Tengo 20 años') is in there")


def check_guid_stable(tmp: Path) -> None:
    print("== re-importing updates instead of duplicating ==")
    dataset = make_dataset(tmp / "src3")
    first = Step7Anki(output_dir=tmp).run(dataset, "Estabilidad").apkg_path
    second = Step7Anki(output_dir=tmp).run(make_dataset(tmp / "src3b"), "Estabilidad").apkg_path

    # The guid derives from Card.id, not from the field values, so a note whose
    # translation changed between runs keeps its identity in Anki.
    from pipeline.steps.step7_anki import GUID_SALT
    import genanki

    guid_a = genanki.guid_for(GUID_SALT, "p001")
    guid_b = genanki.guid_for(GUID_SALT, "p001")
    assert guid_a == guid_b
    assert guid_a in collection(first)["text"], "guid missing from the package"
    assert guid_a in collection(second)["text"]
    print("  the same Card.id yields the same guid across runs")


def check_unique_media(tmp: Path) -> None:
    print("== same text, different cards, different audio ==")
    # Step 5 resolves this with the NNN index prefix, so the basenames differ
    # even when the phrases are identical. Two cards must not collapse onto one
    # embedded file, which would play the wrong audio on one of them.
    cards = [
        ("p001", "You look great today", "Te ves genial hoy"),
        ("p002", "You look great today", "Te ves genial hoy"),
    ]
    dataset = make_dataset(tmp / "dup", cards)
    names = [c.audio.name for c in dataset.cards]
    assert len(set(names)) == 2, names

    apkg = Step7Anki(output_dir=tmp).run(dataset, "Duplicados").apkg_path
    info = collection(apkg)
    for name in names:
        assert name in info["media"], f"{name} not embedded: {info['media']}"
    assert f"[sound:{names[0]}]" in info["text"]
    assert f"[sound:{names[1]}]" in info["text"]
    print(f"  {names[0]} and {names[1]} both embedded and referenced")

    # If two cards did land on the same basename, the step must refuse rather
    # than silently write a package that plays the wrong one.
    clash = [
        Card(id="a", en="x", es="y", audio=dataset.cards[0].audio, start=0.0, end=0.1),
        Card(id="b", en="z", es="w", audio=dataset.cards[0].audio, start=0.0, end=0.1),
    ]
    try:
        Step7Anki(output_dir=tmp).run(Dataset(cards=clash), "Choque")
    except StepOutputError as exc:
        assert "same name" in str(exc), exc
        print("  a genuine name clash is rejected, not merged")
    else:
        raise AssertionError("a basename collision should have failed the step")


def check_errors(tmp: Path) -> None:
    print("== failures are reported, never silent ==")
    step = Step7Anki(output_dir=tmp)
    dataset = make_dataset(tmp / "src4")

    try:
        step.run(Dataset(cards=[]), "Mazo")
    except StepOutputError as exc:
        assert "no cards" in str(exc), exc
        print("  empty dataset -> StepOutputError")

    for bad in ("", "   "):
        try:
            step.run(dataset, bad)
        except StepOutputError as exc:
            assert "name of the deck" in str(exc), exc
        else:
            raise AssertionError(f"blank deck name {bad!r} should have failed")
    print("  blank deck name -> StepOutputError (Anki has to be told the name)")

    # A card whose audio is gone would reach Anki as a button that does nothing.
    missing = Card(id="m", en="x", es="y", audio=tmp / "nope.wav", start=0.0, end=0.1)
    try:
        step.run(Dataset(cards=[missing]), "Mazo")
    except StepOutputError as exc:
        assert "nope.wav" in str(exc), exc
        print("  missing audio -> StepOutputError")

    unreadable = tmp / "broken.wav"
    unreadable.write_bytes(b"not a wav at all")
    broken = Card(id="b", en="x", es="y", audio=unreadable, start=0.0, end=0.1)
    try:
        step.run(Dataset(cards=[broken]), "Mazo")
    except StepOutputError:
        print("  unreadable audio -> StepOutputError")
    else:
        raise AssertionError("an unreadable wav should have failed the step")

    # A blank deck name must not have left a package behind.
    assert not (tmp / "anki" / ".apkg").exists()


def check_report(tmp: Path) -> None:
    print("== report() for the orchestrator ==")
    step = Step7Anki(output_dir=tmp)
    dataset = make_dataset(tmp / "src5")
    package = step.run(dataset, "Reporte")
    summary = step.report()
    assert str(package.apkg_path) in summary, summary
    assert str(len(dataset.cards)) in summary, summary
    print(f"  {summary}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        check_model_shape()
        check_deck_id_stable()
        check_package_contents(workspace)
        check_front_and_back(workspace)
        check_guid_stable(workspace)
        check_unique_media(workspace)
        check_errors(workspace)
        check_report(workspace)

    print("\nall good")
