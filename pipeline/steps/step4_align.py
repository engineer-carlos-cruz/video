"""Step 4 — assign timings to the phrases we want to study.

Takes the transcript of step 3 (words with ``{start, end}``) plus the list of
phrases the user wants, and gives every phrase the time span of its occurrence
in the audio.

The approach is matching over normalized token sequences: both sides are folded
to lowercase, stripped of punctuation and with contractions expanded, then the
phrase is looked up as a contiguous run of tokens inside the word list. The
matching never touches the audio again — it reuses the timestamps step 3 already
produced, so there is no second inference.

* ``start`` is the timestamp of the first matched word and ``end`` the one of
  the last, taken verbatim so the timings stay inside their words.
* The earliest occurrence wins by default; ``first_only=False`` returns them all.
* A phrase with no exact match falls back to a fuzzy comparison over
  ``rapidfuzz`` (C++), which is what absorbs a near-verbatim phrase: a dropped
  contraction, a misheard word. Phrases that still do not land are reported in
  ``self.missing`` rather than silently dropped.

Complexity is O(n + hits·m) per phrase over the n words of the transcript: the
positions of each phrase's first token are looked up once, so only plausible
windows are compared. At the ~100 phrases per video the PRD targets this is
already comfortable; if the list ever grows to thousands, the same loop can be
replaced by a trie (Aho-Corasick) without changing the contract.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.contracts import Phrase, Phrases, Transcript, Word
from pipeline.errors import StepOutputError

# Words, digits and internal apostrophes: "don't" stays one token, and so do
# "well-known" and "1990s". Underscores and punctuation are dropped.
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

_APOSTROPHES = {
    "’": "'",  # right single quotation mark
    "‘": "'",  # left single quotation mark
    "ʼ": "'",  # modifier letter apostrophe
}

# Expanding contractions on *both* sides makes "I'm 20 years old" match a
# transcript that says "I am 20 years old", and the other way round, which is
# the most common source of near-miss phrases.
CONTRACTIONS = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "let's": "let us",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "shan't": "shall not",
}

DEFAULT_FUZZY_THRESHOLD = 85.0

# A contraction turning one token into two makes a 4-word phrase come back as 5
# tokens, so the fuzzy pass also looks at windows one token shorter and longer.
DEFAULT_WINDOW_SLACK = 1


@dataclass(frozen=True)
class MissingPhrase:
    """A phrase that could not be placed in the transcript."""

    index: int
    text: str
    reason: str
    best_score: float | None = None


@dataclass(frozen=True)
class Alignment:
    """What the step found for one input phrase, for reports and tests."""

    index: int
    text: str
    occurrences: tuple[tuple[float, float], ...]
    fuzzy: bool = False
    score: float = 100.0


def _fold(text: str) -> str:
    """Lowercase and normalize apostrophes so both sides compare equal."""
    text = unicodedata.normalize("NFKC", text).lower()
    for fancy, plain in _APOSTROPHES.items():
        text = text.replace(fancy, plain)
    return text


def _expand(token: str) -> list[str]:
    """``"i'm"`` -> ``["i", "am"]``; anything else is returned untouched."""
    expanded = CONTRACTIONS.get(token)
    return expanded.split() if expanded else [token]


def _tokens_of(text: str) -> list[str]:
    """Normalized token sequence of a piece of text."""
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(_fold(text)):
        tokens.extend(_expand(token))
    return tokens


def _sequence(words: Sequence[str]) -> tuple[list[str], list[int]]:
    """Tokens of a run of words plus, for each token, the word it came from.

    A word can yield no token ("—") or more than one, so every token carries an
    explicit back-reference to its word instead of assuming one token per word.
    """
    tokens: list[str] = []
    owners: list[int] = []
    for position, word in enumerate(words):
        for token in _TOKEN_RE.findall(_fold(word)):
            pieces = _expand(token)
            tokens.extend(pieces)
            owners.extend([position] * len(pieces))
    return tokens, owners


def _exact_positions(
    tokens: list[str],
    positions: dict[str, list[int]],
    phrase_tokens: list[str],
) -> list[int]:
    """Start indexes of every exact occurrence of ``phrase_tokens``, in order."""
    length = len(phrase_tokens)
    total = len(tokens)
    hits = []
    for start in positions.get(phrase_tokens[0], ()):
        if start + length <= total and tokens[start : start + length] == phrase_tokens:
            hits.append(start)
    return hits


class Step4Align:
    """Aligns a list of phrases against the word timings of a ``Transcript``."""

    def __init__(
        self,
        allow_fuzzy: bool = True,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        first_only: bool = True,
        window_slack: int = DEFAULT_WINDOW_SLACK,
    ) -> None:
        if not 0.0 <= fuzzy_threshold <= 100.0:
            raise StepOutputError(
                f"fuzzy_threshold must be a percentage, got {fuzzy_threshold!r}"
            )
        if window_slack < 0:
            raise StepOutputError(f"window_slack cannot be negative, got {window_slack}")
        self.allow_fuzzy = allow_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.first_only = first_only
        self.window_slack = window_slack
        self.alignments: list[Alignment] = []
        self.missing: list[MissingPhrase] = []

    def run(self, transcript: Transcript, phrases: Sequence[str] | str) -> Phrases:
        if transcript is None or transcript.is_empty():
            raise StepOutputError("Step 4 received an empty transcript")
        wanted = [phrases] if isinstance(phrases, str) else list(phrases)
        if not wanted:
            raise StepOutputError("Step 4 received no phrases to align")

        words = list(transcript.words)
        tokens, owners = _sequence([word.word for word in words])
        if not tokens:
            raise StepOutputError(
                "Step 4 could not tokenize the transcript into any word"
            )

        positions: dict[str, list[int]] = defaultdict(list)
        for index, token in enumerate(tokens):
            positions[token].append(index)
        vocabulary = list(positions)

        self.alignments = []
        self.missing = []
        matched: list[Phrase] = []

        for index, text in enumerate(wanted):
            phrase_tokens = _tokens_of(text)
            counter = len(matched) + 1
            if not phrase_tokens:
                self._report(index, text, "the phrase has no words to match", None)
                continue

            exact = _exact_positions(tokens, positions, phrase_tokens)
            fuzzy = False
            score = 100.0
            best_score: float | None = None
            windows = [(start, len(phrase_tokens)) for start in exact]

            if not windows and self.allow_fuzzy:
                best, best_score = self._fuzzy_window(
                    tokens, positions, vocabulary, phrase_tokens
                )
                if best is not None:
                    start, length, score = best
                    windows = [(start, length)]
                    fuzzy = True

            if not windows:
                if not self.allow_fuzzy:
                    self._report(index, text, "no occurrence in the transcript", None)
                elif best_score is None:
                    self._report(
                        index,
                        text,
                        "none of its words look like anything in the transcript",
                        None,
                    )
                else:
                    self._report(
                        index,
                        text,
                        f"best fuzzy match {best_score:.0f}% is below the "
                        f"{self.fuzzy_threshold:.0f}% threshold",
                        best_score,
                    )
                continue

            if self.first_only:
                windows = windows[:1]

            occurrences = tuple(
                self._span(words, owners, start, length) for start, length in windows
            )
            self.alignments.append(
                Alignment(
                    index=index,
                    text=text,
                    occurrences=occurrences,
                    fuzzy=fuzzy,
                    score=round(score, 1),
                )
            )
            for occurrence, (start, end) in enumerate(occurrences, start=1):
                identifier = f"p{counter:03d}"
                if len(occurrences) > 1:
                    identifier = f"{identifier}.{occurrence}"
                matched.append(
                    Phrase(id=identifier, text=text.strip(), start=start, end=end)
                )

        if not matched:
            raise StepOutputError(
                f"Step 4 could not place any of the {len(wanted)} phrases in the "
                f"transcript ({len(self.missing)} unmatched, "
                f"{len(words)} words transcribed)"
            )
        return Phrases(phrases=matched)

    def report(self) -> str:
        """Human-readable summary of the last run, for the orchestrator."""
        total = len(self.alignments) + len(self.missing)
        lines = [f"aligned {len(self.alignments)}/{total} phrases"]
        for miss in self.missing:
            lines.append(f"  not found [{miss.index}]: {miss.text!r} ({miss.reason})")
        return "\n".join(lines)

    def _report(
        self, index: int, text: str, reason: str, score: float | None
    ) -> None:
        self.missing.append(
            MissingPhrase(
                index=index,
                text=text,
                reason=reason,
                best_score=None if score is None else round(score, 1),
            )
        )

    def _fuzzy_window(
        self,
        tokens: list[str],
        positions: dict[str, list[int]],
        vocabulary: list[str],
        phrase_tokens: list[str],
    ) -> tuple[tuple[int, int, float] | None, float | None]:
        """Best window above the threshold, plus the best score seen either way.

        Candidates are anchored on the phrase's first token, so the scan costs
        what the occurrences of that token cost instead of the whole transcript.
        When the first word itself was misheard, the closest vocabulary entry
        anchors the search instead.
        """
        modules = self._load_rapidfuzz()
        anchor = phrase_tokens[0]
        candidates = positions.get(anchor)
        if not candidates:
            nearest = modules["process"].extractOne(
                anchor, vocabulary, score_cutoff=self.fuzzy_threshold / 2.0
            )
            candidates = positions.get(nearest[0], []) if nearest else []

        needle = " ".join(phrase_tokens)
        total = len(tokens)
        best: tuple[int, int, float] | None = None
        best_score: float | None = None

        for start in candidates:
            for length in range(
                max(1, len(phrase_tokens) - self.window_slack),
                min(total - start, len(phrase_tokens) + self.window_slack) + 1,
            ):
                window = tokens[start : start + length]
                score = modules["fuzz"].ratio(needle, " ".join(window))
                if best_score is None or score > best_score:
                    best_score = score
                if score >= self.fuzzy_threshold and (best is None or score > best[2]):
                    best = (start, length, score)
        return best, best_score

    @staticmethod
    def _load_rapidfuzz() -> dict:
        """Import RapidFuzz lazily, so the exact path needs no dependency."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise StepOutputError(
                "rapidfuzz is not installed in this environment "
                "(uv pip install rapidfuzz)"
            ) from exc
        return {"fuzz": fuzz, "process": process}

    @staticmethod
    def _span(
        words: list[Word], owners: list[int], start: int, length: int
    ) -> tuple[float, float]:
        """Times of the words backing a matched token window.

        Coerced to plain ``float``: step 3 hands over the numpy scalars that
        faster-whisper produces, and the timings are serialized further down.
        """
        first = words[owners[start]]
        last = words[owners[start + length - 1]]
        begin = max(0.0, round(float(first.start), 3))
        return begin, max(begin, round(float(last.end), 3))


__all__ = ["Alignment", "MissingPhrase", "Step4Align"]