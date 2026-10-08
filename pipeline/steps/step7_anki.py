"""Step 7 — export the dataset as an Anki package.

Takes the ``Dataset`` of step 6 (per phrase: English, idiomatic Spanish and the
audio fragment) and writes a single ``.apkg`` holding every card at once, so
importing it into Anki is one operation instead of one per phrase.

This is the **only step coupled to Anki**, and by design it is replaceable: the
``Dataset`` it consumes carries no trace of any consumer, so a CSV exporter or a
web app can read the very same output without touching steps 1–6.

Deck layout
-----------
One fixed note model, ``ES-EN Audio``, with three fields: ``Español``,
``Inglés`` and ``Audio``. Spanish is the front because that is what the user
recalls; the back shows the English sentence plus the play button, which is
where the audio actually matters.

The model id is a module constant rather than something generated per run. That
is the whole point of the PRD's "id estable para reutilizarlo entre
importaciones": if the id changed on every execution, Anki would see a brand new
note type each time instead of reusing this one. The deck id is derived from the
deck name the same way, so re-importing under the same name adds the notes to
the existing deck instead of creating a duplicate one.

Audio
-----
Fragments are embedded in the package and referenced as ``[sound:<name>.wav]``,
which is what renders Anki's play button. The name used inside the package is
the fragment's own basename. Step 5 already names those ``NNN_<slug>.wav`` with
``NNN`` the position of the phrase in ``Phrases``, so the basenames are unique
even when two phrases share their text. Reusing them as-is means the wavs are
not copied or renamed, and ``Card.audio`` keeps pointing at a real file on disk.

Every ``Card``'s audio is validated with ``probe_wav`` before anything is
written. A card whose audio is missing or unreadable would reach Anki as a
button that silently does nothing, which is exactly the failure that is
invisible until study time.

Decoupling note
---------------
``guid`` is built with ``genanki.guid_for`` rather than invented here, so notes
carry the base91 form Anki expects. It is derived from ``Card.id``, which is
stable across runs, so re-importing the same dataset updates the existing notes
instead of duplicating them.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import genanki

from pipeline.contracts import AnkiPackage, Card, Dataset
from pipeline.errors import StepOutputError
from pipeline.media import probe_wav

DEFAULT_OUTPUT_DIR = Path("output")

# Packages live one level below the step outputs, mirroring where step 5 keeps
# the fragments.
ANKI_DIRNAME = "anki"

# The PRD's dedicated fixed model, with a stable id so Anki reuses the note type
# across imports instead of piling up copies of it.
MODEL_ID = 1729480001
MODEL_NAME = "ES-EN Audio"

# Namespace for the per-card guid. genanki's own hashing is used to build it;
# this salt keeps the value from being confused with a guid derived from the
# note's field values.
GUID_SALT = "video-es-en-audio"

FIELD_SPANISH = "Español"
FIELD_ENGLISH = "Inglés"
FIELD_AUDIO = "Audio"

FIELD_NAMES = [FIELD_SPANISH, FIELD_ENGLISH, FIELD_AUDIO]

# Spanish first: it is what the user is asked to recall. The English sentence
# and the play button together form the answer.
MODEL_CSS = (
    ".card {"
    "  font-family: -apple-system, 'Segoe UI', sans-serif;"
    "  font-size: 20px;"
    "  text-align: center;"
    "  color: #1a1a1a;"
    "  background-color: #f7f7f5;"
    "}"
    ".es { font-size: 26px; margin-bottom: 12px; }"
    ".en { font-size: 20px; color: #444; }"
)

_ILLEGAL_FILENAME_CHARS = re.compile(r"[^\w\s-]", re.UNICODE)
_SPACES = re.compile(r"[\s_-]+", re.UNICODE)


def _slugify(value: str, max_length: int = 60) -> str:
    """Return a lowercase, underscore-separated fragment of ``value``.

    Private copy, as in step 5: a step must not reach into another step's
    internals, only into the contracts.
    """
    text = unicodedata.normalize("NFKC", value).lower()
    text = _ILLEGAL_FILENAME_CHARS.sub(" ", text)
    text = _SPACES.sub("_", text).strip("_")
    return text[:max_length].strip("_") or "deck"


def _deck_id(name: str) -> int:
    """Return a stable Anki deck id derived from ``name``.

    Anki identifies a deck by this id, so it has to be the same every time the
    same name is used. genanki would otherwise generate a fresh random one per
    run, which makes every import create a duplicate deck instead of adding to
    the existing one.
    """
    digest = hashlib.sha256(name.strip().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def build_model() -> genanki.Model:
    """Return the fixed ``ES-EN Audio`` note model.

    genanki assigns the ``ord`` of each field itself when writing the package,
    so the fields go in as plain names.
    """
    return genanki.Model(
        MODEL_ID,
        MODEL_NAME,
        fields=[{"name": name} for name in FIELD_NAMES],
        templates=[
            {
                "name": "Recognise",
                "qfmt": '<div class="es">{{Español}}</div>',
                "afmt": (
                    '{{FrontSide}}<hr id="answer">'
                    '<div class="en">{{Inglés}}</div>'
                    '<div>{{Audio}}</div>'
                ),
            }
        ],
        css=MODEL_CSS,
    )


class Step7Anki:
    """Writes a ``Dataset`` out as a ready-to-import ``.apkg``."""

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.notes: list[Card] = []
        self.apkg_path: Path | None = None

    def run(self, dataset: Dataset, deck_name: str) -> AnkiPackage:
        if dataset is None or dataset.is_empty():
            raise StepOutputError("Step 7 received no cards to export")

        if deck_name is None or not deck_name.strip():
            raise StepOutputError(
                "Step 7 needs the name of the deck to create; Anki adds the cards "
                "to an existing deck with that name or creates one automatically"
            )

        name = deck_name.strip()
        media = self._collect_media(dataset.cards)
        notes = self._build_notes(dataset.cards)

        directory = self.output_dir / ANKI_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        # The .apkg filename is still an open question in the PRD (§11); this is a
        # provisional convention so re-running overwrites rather than piles up.
        path = directory / f"{_slugify(name)}.apkg"

        deck = genanki.Deck(_deck_id(name), name)
        deck.add_model(build_model())
        for note in notes:
            deck.add_note(note)
        package = genanki.Package(deck)
        package.media_files = [card.audio for card in dataset.cards]
        package.write_to_file(str(path))

        result = AnkiPackage(apkg_path=path)
        # The contract already detects a missing or empty package; checking here
        # means the failure surfaces in this step rather than at import time in
        # Anki.
        if result.is_empty():
            raise StepOutputError(f"Step 7 wrote no usable package at {path}")

        self.notes = list(dataset.cards)
        self.apkg_path = path
        return result

    def report(self) -> str:
        """Human-readable summary of the last run, for the orchestrator."""
        target = self.apkg_path if self.apkg_path is not None else "nothing exported"
        return f"exported {len(self.notes)} cards to {target}"

    def _collect_media(self, cards: list[Card]) -> dict[str, Path]:
        """Validate every fragment and map basename -> path.

        Two cards landing on the same basename is the one case that could make
        the package play the wrong audio, so it is rejected rather than merged.
        Step 5 makes it impossible today (the index prefix keeps names unique),
        but the guarantee belongs to this step's output, not to step 5's.
        """
        media: dict[str, Path] = {}
        for card in cards:
            probe_wav(card.audio)
            key = card.audio.name
            if key in media:
                raise StepOutputError(
                    f"Step 7 found two cards whose audio has the same name "
                    f"({key}: {media[key]} and {card.audio}); the package could "
                    f"play the wrong one, so it is not written"
                )
            media[key] = card.audio
        return media

    def _build_notes(self, cards: list[Card]) -> list[genanki.Note]:
        model = build_model()
        notes = []
        for card in cards:
            notes.append(
                genanki.Note(
                    model=model,
                    fields=[
                        card.es,
                        card.en,
                        f"[sound:{card.audio.name}]",
                    ],
                    # Derived from Card.id, which survives across runs, so
                    # re-importing updates the notes instead of duplicating them.
                    guid=genanki.guid_for(GUID_SALT, card.id),
                    tags=["es-en-audio"],
                )
            )
        return notes
