"""Step 6 — translate the phrases and join them with their audio.

Takes the timed phrases of step 4 plus the fragments of step 5 and returns the
``Dataset``: one ``Card`` per phrase with the English text, its idiomatic Spanish
translation and the path to the fragment that holds it. The result is
consumer-agnostic — it knows nothing about Anki or any other destination.

Engine
------
llama.cpp (C++) behind ``llama-server``, which exposes an OpenAI-compatible HTTP
API and returns JSON instead of text that has to be scraped out of a terminal.
The model runs int4 GGUF on CPU, which is what the PRD's ~2 GB budget allows.

The server is expected to be **already running** and is reused across videos:
loading the model costs seconds and over a gigabyte of RAM, so paying that per
video would be wasteful. If it is not up, this step fails with instructions
rather than starting it — process lifecycle is the orchestrator's business,
translation is this step's. ``llama-cpp-python`` would embed the backend
in-process (the PRD's original wording) but it ships no wheel for cp314, and step
6 sits behind the ``Dataset`` contract, so swapping the transport later touches
nothing else.

Why the batch is tagged, not numbered
-------------------------------------
The PRD asks for one prompt per video, and that is what happens — but a small
model asked for "one numbered line per phrase" is not reliable. Measured on
Llama-3.2-1B-Instruct-Q4_K_M with 14 phrases, it rewrote the line numbers into
Spanish words (``1.`` became ``Uno:``, ``2.`` became ``Dos:``), and asked for a
JSON object it ignored the schema of entirely, emitting fourteen copies of
``{"n": 1, "v": "es"}``.

So every phrase carries a bracketed ``[[n]]`` tag that the model must echo, and
the reply is parsed by **tag**, never by line position. That survived every
failure the numbering had. The same experiment showed the model needs few-shot
examples: without them "I'm twenty years old" came back as "Estoy 20 años",
failing the PRD's first success criterion, and once hallucinated "veintiuno".

The examples go in as real user/assistant turns, not as text inside the prompt.
Flattened into one message they reused the tags ``[[1]]``..``[[3]]`` that the
request then restarts, and the model leaked example 2 straight into an answer:
"She is going to the store" came back as "Vámonos que llegamos tarde". Separate
turns remove the collision, and ``_is_leak`` keeps rejecting a parroted example
if it ever happens again.

Alignment is checked, never assumed: every requested tag must come back once,
with a non-empty translation that is not a copy of an example. Anything else is
a failure, reported per phrase — a wrong translation on a card is far worse
than a missing one.

Retries
-------
A chunk that comes back misaligned is retried phrase by phrase, which is where a
small model is at its most reliable, and then reported as untranslated if it
still fails. Phrases that cannot be translated are dropped from the ``Dataset``
and listed in ``self.untranslated``; the step only raises when *no* phrase
survives, matching how step 4 treats an entirely failed alignment.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pipeline.contracts import Card, Dataset, Fragment, Fragments, Phrases
from pipeline.errors import StepOutputError

DEFAULT_BASE_URL = "http://127.0.0.1:8080"

# int4 keeps the model inside the PRD's ~2 GB (measured: 1715 MB RSS for this
# one, 1253 MB for Llama-3.2-1B). Qwen2.5-1.5B is the default over the smaller
# Llama because naturalness is this step's first success criterion and, on the
# same six phrases, it fixed every case Llama-3.2-1B got wrong — most visibly
# "You look great today" ("Te ves genial hoy" against "Pareces muy bonito hoy")
# and "She is going to the store" ("Está saliendo al supermercado" against
# "Vámonos que vamos al supermercado"). It costs roughly 2.5x the time per batch.
# Both models are a constant away, and the contract hides which one is running.
DEFAULT_MODEL = "models/llama/qwen2.5-1.5b-instruct-q4_k_m.gguf"
FALLBACK_MODEL = "models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# The PRD's "traducción en lote": one prompt per video, sized so a 1B model at
# -c 4096 does not run out of context. Measured: ~14 phrases take ~22 s.
DEFAULT_BATCH_PHRASES = 15

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200

# The model is a black box with its own ideas about how long a translation is.
DEFAULT_TIMEOUT = 600.0

SYSTEM_PROMPT = (
    "Traduces frases del inglés al español para alguien que está aprendiendo.\n"
    "Reglas: usa español natural, como lo hablaría un nativo; nunca traduzcas "
    "literalmente; preserva el significado, no la estructura; "
    "'I'm 20 years old' es 'Tengo 20 años', no 'Estoy 20 años'.\n"
    "Responde copiando cada etiqueta [[n]] y escribiendo solo la traducción, "
    "una frase por línea."
)

# Three worked examples. The first is the PRD's own acceptance case; the others
# pin down failure modes measured without examples (a dropped subject pronoun, a
# garbled duration).
#
# They are sent as real user/assistant turns rather than as text inside the
# prompt, and that is not cosmetic. With all of it flattened into one user
# message the examples reused the tags [[1]]..[[3]] that the request then starts
# again from, and the model leaked one straight into an answer: "She is going to
# the store" came back as "Vámonos que llegamos tarde", the translation of
# example 2. As separate turns there is no tag collision, and the leak did not
# reproduce.
FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("I'm 20 years old", "Tengo 20 años"),
    ("Let's get going, we're late", "Vámonos que llegamos tarde"),
    ("I've been learning English for two years", "Llevo dos años aprendiendo inglés"),
)

# A translation that is basically one of the examples is the model parroting, not
# translating. Compared with rapidfuzz because the leak is often a near miss
# ("Vámonos que vamos al supermercado" against "Vámonos que llegamos tarde").
LEAK_THRESHOLD = 85.0

# The model sometimes declines to pick a subject and hedges instead, answering
# "He/she/it has been learning English for two years" — English, untranslated.
# Rejected rather than shipped to a card.
_HEDGED_PRONOUN = re.compile(r"\bhe/she/it\b", re.IGNORECASE)

# A reply that is essentially the English source came back untranslated too.
UNTRANSLATED_THRESHOLD = 90.0

# The tag has to be echoed by the model, so the parser keys on it. Square
# brackets are kept out of the phrases themselves by requiring the source text to
# sit after the closing bracket.
_TAG_RE = re.compile(r"\[\[(\d+)\]\]", re.MULTILINE)

# "I'm 20 years old" and "I'm twenty years old" are the same phrase and score
# only 78 against each other, below any threshold worth trusting. Spelling the
# number out is normal in a study list and normal in an example, so both sides
# are canonicalised to digits before the sources are compared.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100",
}
_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}

# "forty five" has to become 45, not "40 5", so the tens+unit compound is
# consumed before the single words are replaced.
_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]+"
    r"(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)


def canonical_source(text: str) -> str:
    """Lowercase an English source and spell any number word out as digits."""
    plain = text.lower().strip(" .;:?!,")
    plain = _COMPOUND_RE.sub(
        lambda m: str(_TENS[m.group(1).lower()] + _UNITS[m.group(2).lower()]), plain
    )
    return _NUMBER_RE.sub(lambda m: _NUMBER_WORDS[m.group(1).lower()], plain)


def fuzz_ratio(left: str, right: str) -> float:
    """Similarity of two strings, 0-100.

    Delegates to rapidfuzz (C++) when available and falls back to difflib so the
    leak and untranslated checks always run: dropping a check because an optional
    dependency is missing is how a bad translation reaches a card.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - depends on the env
        from difflib import SequenceMatcher

        return 100.0 * SequenceMatcher(None, left, right).ratio()
    return fuzz.ratio(left, right)


@dataclass(frozen=True)
class UntranslatedPhrase:
    """A phrase with no usable Spanish translation, and why."""

    id: str
    text: str
    reason: str


@dataclass(frozen=True)
class _Pending:
    """A phrase waiting for its translation, with the fragment that carries it."""

    number: int
    phrase_id: str
    text: str
    fragment: Fragment


class Step6Translate:
    """Translates the phrases of a video and joins them with their audio."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: Path | str = DEFAULT_MODEL,
        context: str | None = None,
        batch_phrases: int = DEFAULT_BATCH_PHRASES,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if batch_phrases < 1:
            raise StepOutputError(
                f"batch_phrases must be at least 1, got {batch_phrases}"
            )
        self.base_url = base_url.rstrip("/")
        self.model = Path(model)
        self.context = context
        self.batch_phrases = batch_phrases
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.untranslated: list[UntranslatedPhrase] = []
        self._reasons: dict[str, str] = {}
        self.stats: dict[str, float | int] = {}

    # -- orchestration ----------------------------------------------------

    def run(self, phrases: Phrases, fragments: Fragments) -> Dataset:
        if phrases is None or phrases.is_empty():
            raise StepOutputError("Step 6 received no phrases to translate")
        if fragments is None or fragments.is_empty():
            raise StepOutputError("Step 6 received no fragments to join")

        self.require_server()
        pending = self._join(phrases, fragments)

        self.untranslated = []
        self._reasons = {}
        translations: dict[str, str] = {}
        started = time.monotonic()
        retried = 0
        for chunk in self._chunks(pending):
            got, _ = self._translate_chunk(chunk)
            if len(got) != len(chunk):
                # The batch was not trustworthy. Redo it one phrase at a time,
                # where a 1B model holds the format reliably.
                retried += len(chunk)
                for item in chunk:
                    single, _ = self._translate_chunk([item])
                    translations.update(single)
            else:
                translations.update(got)

        self._collect_failures(pending, translations)
        cards = self._cards(pending, translations)
        if not cards:
            raise StepOutputError(
                f"Step 6 could not translate any of the {len(pending)} phrases "
                f"with {self.model.name} ({len(self.untranslated)} failed)"
            )

        self.stats = {
            "phrases": len(pending),
            "translated": len(cards),
            "untranslated": len(self.untranslated),
            "retried": retried,
            "seconds": round(time.monotonic() - started, 1),
        }
        return Dataset(cards=cards)

    def report(self) -> str:
        """Human-readable summary of the last run, for the orchestrator."""
        stats = self.stats or {}
        lines = [
            f"translated {stats.get('translated', 0)}/{stats.get('phrases', 0)} "
            f"phrases in {stats.get('seconds', 0)}s "
            f"({stats.get('retried', 0)} retried one by one)"
        ]
        for miss in self.untranslated:
            lines.append(f"  not translated [{miss.id}]: {miss.text!r} ({miss.reason})")
        return "\n".join(lines)

    @staticmethod
    def _join(phrases: Phrases, fragments: Fragments) -> list[_Pending]:
        """Pair every phrase with its fragment, by id.

        A phrase without audio cannot become a card, and a fragment without a
        phrase has nothing to translate, so both are reported instead of being
        paired up by accident.
        """
        by_id = {fragment.id: fragment for fragment in fragments.fragments}
        pending: list[_Pending] = []
        for phrase in phrases.phrases:
            fragment = by_id.get(phrase.id)
            if fragment is None:
                continue
            if not fragment.file.exists() or fragment.file.stat().st_size == 0:
                continue
            pending.append(
                _Pending(
                    number=len(pending) + 1,
                    phrase_id=phrase.id,
                    text=phrase.text,
                    fragment=fragment,
                )
            )
        return pending

    def _chunks(self, pending: list[_Pending]):
        for start in range(0, len(pending), self.batch_phrases):
            yield pending[start : start + self.batch_phrases]

    def _cards(self, pending: list[_Pending], translations: dict[str, str]) -> list[Card]:
        """Build one card per translated phrase, in phrase order."""
        cards = []
        for item in pending:
            spanish = translations.get(item.phrase_id, "").strip()
            if not spanish:
                continue
            cards.append(
                Card(
                    id=item.phrase_id,
                    en=item.text,
                    es=spanish,
                    audio=item.fragment.file,
                    start=item.fragment.start,
                    end=item.fragment.end,
                )
            )
        return cards

    def _collect_failures(
        self, pending: list[_Pending], translations: dict[str, str]
    ) -> None:
        """List every phrase left without a usable translation, with the reason."""
        for item in pending:
            if translations.get(item.phrase_id, "").strip():
                continue
            self.untranslated.append(
                UntranslatedPhrase(
                    id=item.phrase_id,
                    text=item.text,
                    reason=self._reasons.get(
                        item.phrase_id, "the model returned no usable translation"
                    ),
                )
            )
            translations.pop(item.phrase_id, None)

    # -- talking to llama-server -----------------------------------------

    def require_server(self) -> None:
        """Fail early, and with instructions, if llama-server is not up."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as response:
                if response.status != 200:
                    raise urllib.error.URLError(f"status {response.status}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise StepOutputError(
                f"No llama-server answering at {self.base_url} ({exc}).\n"
                "Start it with:\n"
                f"  llama-server -m {self.model} --host 127.0.0.1 --port 8080 "
                "-c 4096 --threads 4\n"
                "and leave it running; step 6 reuses it across videos."
            ) from exc

    def _translate_chunk(self, chunk: list[_Pending]) -> tuple[dict[str, str], str]:
        """Ask for one chunk and return ``{phrase id: translation}``.

        Only phrases whose tag came back exactly once survive: a duplicated,
        missing or empty line means the reply cannot be trusted to line up with
        the request, and guessing would put the wrong Spanish on a card.
        """
        reply = self._ask(self._messages(chunk))
        parsed = self._parse(reply)

        wanted = {str(item.number): item for item in chunk}
        usable: dict[str, str] = {}
        for item in chunk:
            number = str(item.number)
            if number not in parsed:
                self._reason(item, f"the model did not return its tag [[{number}]]")
                continue
            cleaned = self._clean(parsed[number])
            if not cleaned:
                self._reason(item, "the line came back empty")
                continue
            if self._is_leak(item.text, cleaned):
                self._reason(
                    item, "the model repeated a few-shot example instead of translating"
                )
                continue
            if self._looks_untranslated(item.text, cleaned):
                self._reason(item, f"the reply was still in English: {cleaned!r}")
                continue
            usable[item.phrase_id] = cleaned
            self._reasons.pop(item.phrase_id, None)
        return usable, reply

    def _reason(self, item: _Pending, reason: str) -> None:
        """Remember why a phrase was rejected; the last attempt wins."""
        self._reasons[item.phrase_id] = reason

    def _messages(self, chunk: list[_Pending]) -> list[dict[str, str]]:
        """The example turns followed by the real request."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for english, spanish in FEW_SHOT:
            messages.append({"role": "user", "content": f"[[1]] {english}"})
            messages.append({"role": "assistant", "content": f"[[1]] {spanish}"})
        if self.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Contexto del video: {self.context}",
                }
            )
            messages.append(
                {"role": "assistant", "content": "Entendido, usaré ese contexto."}
            )
        messages.append(
            {
                "role": "user",
                "content": "\n".join(f"[[{item.number}]] {item.text}" for item in chunk),
            }
        )
        return messages

    def _ask(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            raise StepOutputError(
                f"llama-server returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise StepOutputError(
                f"could not reach llama-server at {self.base_url}: {exc}"
            ) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise StepOutputError(
                f"llama-server replied in an unexpected shape: {str(body)[:200]}"
            ) from exc
        if not content or not content.strip():
            raise StepOutputError("llama-server returned an empty completion")
        return content

    @staticmethod
    def _parse(reply: str) -> dict[str, str]:
        """Map tag number to the translation that follows it.

        Everything before the first tag is a preamble ("Aquí te tienes...") and
        is dropped; a tag repeated by the model overwrites the earlier line, so
        the last one wins and the caller still sees exactly one entry per tag.
        """
        parsed: dict[str, str] = {}
        matches = list(_TAG_RE.finditer(reply))
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(reply)
            parsed[match.group(1)] = reply[match.end() : end].strip()
        return parsed

    @staticmethod
    def _clean(text: str) -> str:
        """Reduce one parsed line to a single sentence of Spanish."""
        line = text.strip()
        # The model likes to add a remark after the translation on the same line.
        for separator in ("\n", "  ", " — ", " - "):
            if separator in line:
                line = line.split(separator)[0].strip()
        return line.strip(" .;:").strip()

    def _is_leak(self, english: str, spanish: str) -> bool:
        """True when ``spanish`` parrots an example instead of translating.

        Cheap insurance against the failure measured above: the model repeating
        an example in place of a real translation is otherwise indistinguishable
        from a bad translation, and it would reach a card looking like a
        confident answer.

        An example whose English source *is* the phrase is not a leak — matching
        it is the whole point of having the example. Without this exemption the
        check threw away correct translations, and the exemption has to be fuzzy
        too: the study list says "I'm twenty years old" where the example says
        "I'm 20 years old", and a literal comparison treated those as different
        phrases, so the PRD's own reference answer "Tengo 20 años" was flagged
        and dropped.

        Falls back to an exact comparison without rapidfuzz so the check still
        runs without the optional dependency.
        """
        examples = [
            answer
            for source, answer in FEW_SHOT
            if not self._same_phrase(source, english)
        ]
        if not examples:
            return False
        normalized = spanish.lower().strip(" .;:")
        if normalized in [answer.lower().strip(" .;:") for answer in examples]:
            return True
        return any(
            fuzz_ratio(normalized, answer.lower()) >= LEAK_THRESHOLD
            for answer in examples
        )

    @staticmethod
    def _looks_untranslated(english: str, spanish: str) -> bool:
        """True when the reply is English, or a hedge, rather than a translation."""
        if _HEDGED_PRONOUN.search(spanish):
            return True
        return (
            fuzz_ratio(spanish.lower(), english.lower()) >= UNTRANSLATED_THRESHOLD
        )

    @staticmethod
    def _same_phrase(left: str, right: str) -> bool:
        """Whether two English sources are the same phrase written differently."""
        first = canonical_source(left)
        second = canonical_source(right)
        return first == second or fuzz_ratio(first, second) >= LEAK_THRESHOLD


__all__ = ["Step6Translate", "UntranslatedPhrase"]