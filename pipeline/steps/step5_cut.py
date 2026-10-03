"""Step 5 — cut one audio fragment per phrase.

Takes the cleaned wav of step 2 plus the timed phrases of step 4 and writes one
``.wav`` per phrase, with ~100 ms of margin on both sides so the fragment never
clips the attack or the release of the first and last phoneme. Cuts overlap
freely: each fragment is independent, and nothing else in the pipeline depends on
them not overlapping.

The whole run is a **single FFmpeg invocation** (C) with one output file per
fragment, which is what keeps ~100 cuts cheap: one demux, one decode of the
source, no process per phrase.

Why ``atrim`` and not ``-ss``/``-to``
-----------------------------------
The PRD suggests ``-ss``/``-to``. Measured on this machine (FFmpeg 8.0.1) they
are only trustworthy when the audio is re-encoded: with ``-c:a copy`` a request
for 1.0s..3.0s of a 5s file came back with 2.048s of audio, and 0.2537..0.7537
came back with 0.512s instead of 0.500s. Even when they do work, the result
depends on how FFmpeg rounds the timestamps.

``atrim=start_sample=..:end_sample=..`` works in sample indices instead, so the
bounds are computed here in Python (``round(seconds * sample_rate)``) and the
filter only has to obey them. Verified byte-exact against the source bytes at
48 kHz/stereo and 44.1 kHz/mono, so criterion 2 ("no temporal drift") holds
literally rather than approximately.

Bounds are clamped in Python *before* the filtergraph is built. FFmpeg treats an
``atrim`` that starts past EOF as a hard error that fails the whole invocation
(exit 234) while printing one error per affected output, so an out-of-range
phrase must never reach the command line.

Documented alternative
----------------------
Slicing the PCM directly in Python (``wave``/numpy, ``index = round(seconds *
sample_rate)``) is byte-exact too and avoids the subprocess entirely, at the cost
of loading and rewriting the file once per fragment. Not implemented: 100 cuts
take ~0.23s in a single FFmpeg run here, which leaves nothing to gain, and the
FFmpeg path is the one the PRD nominates. Revisit if the fragment count grows by
an order of magnitude.

Decoupling note
---------------
``slugify`` is reimplemented privately below rather than imported from
``step1_download``: a step must not reach into another step's internals, only
into the contracts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pipeline.contracts import CleanedAudio, Fragment, Fragments, Phrases
from pipeline.errors import StepOutputError
from pipeline.media import verify_wav

DEFAULT_OUTPUT_DIR = Path("output")

# Fragments live one level below the step outputs, in a per-video folder, so a
# hundred wavs never pile up in a single directory.
FRAGMENTS_DIRNAME = "fragments"

# The PRD's "~100 ms before and after". A phrase cut exactly at a zero-length
# span would still get 200 ms of audio, which is why the margin is generous
# compared with a single phoneme.
DEFAULT_MARGIN_SECONDS = 0.1

# A batch is one FFmpeg invocation. ~100 phrases per video fit in one command
# (argv stays around 12 kB, far below any limit); batching only exists so the
# command line cannot grow without bound if the phrase count ever explodes.
DEFAULT_BATCH_SIZE = 64

_ILLEGAL_FILENAME_CHARS = re.compile(r"[^\w\s-]", re.UNICODE)
_SPACES = re.compile(r"[\s_-]+", re.UNICODE)


def _slugify(value: str, max_length: int = 60) -> str:
    """Return a lowercase, underscore-separated fragment of ``value``."""
    text = unicodedata.normalize("NFKC", value).lower()
    text = _ILLEGAL_FILENAME_CHARS.sub(" ", text)
    text = _SPACES.sub("_", text).strip("_")
    return text[:max_length].strip("_") or "phrase"


@dataclass(frozen=True)
class SkippedPhrase:
    """A phrase that could not be turned into a fragment, and why."""

    index: int
    id: str
    text: str
    reason: str


@dataclass(frozen=True)
class _Cut:
    """A fragment to produce, already resolved to sample indices."""

    index: int
    id: str
    text: str
    path: Path
    start_frame: int
    end_frame: int
    sample_rate: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start(self) -> float:
        return self.start_frame / self.sample_rate

    @property
    def end(self) -> float:
        return self.end_frame / self.sample_rate


class Step5Cut:
    """Cuts one fragment per phrase out of a ``CleanedAudio``."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        margin: float = DEFAULT_MARGIN_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float | None = None,
    ) -> None:
        if margin < 0:
            raise StepOutputError(f"margin cannot be negative, got {margin}")
        if batch_size < 1:
            raise StepOutputError(f"batch_size must be at least 1, got {batch_size}")
        self.output_dir = Path(output_dir)
        self.margin = margin
        self.batch_size = batch_size
        self.timeout = timeout
        self.fragments: list[Fragment] = []
        self.skipped: list[SkippedPhrase] = []

    def run(self, cleaned: CleanedAudio, phrases: Phrases) -> Fragments:
        if cleaned.is_empty():
            raise StepOutputError(f"Step 5 received an empty input: {cleaned.file}")
        if phrases is None or phrases.is_empty():
            raise StepOutputError("Step 5 received no phrases to cut")

        self._require_ffmpeg()
        source = verify_wav(cleaned.file)

        directory = self._fragment_dir(cleaned)
        directory.mkdir(parents=True, exist_ok=True)

        self.skipped = []
        cuts: list[_Cut] = []
        for index, phrase in enumerate(phrases.phrases, start=1):
            cut = self._plan(index, phrase, directory, source)
            if cut is not None:
                cuts.append(cut)

        if not cuts:
            raise StepOutputError(
                f"Step 5 could not cut any of the {len(phrases.phrases)} phrases "
                f"out of {cleaned.file} "
                f"({len(self.skipped)} outside the audio or empty)"
            )

        for batch in self._batches(cuts):
            self._run_batch(cleaned.file, batch, source)

        fragments = [
            Fragment(id=cut.id, file=cut.path, start=cut.start, end=cut.end)
            for cut in cuts
        ]
        result = Fragments(fragments=fragments)
        if result.is_empty():
            raise StepOutputError(f"Step 5 produced no usable fragments from {cleaned.file}")
        self.fragments = fragments
        return result

    def report(self) -> str:
        """Human-readable summary of the last run, for the orchestrator."""
        lines = [f"cut {len(self.fragments)} fragments"]
        for miss in self.skipped:
            lines.append(f"  skipped [{miss.index}]: {miss.text!r} ({miss.reason})")
        return "\n".join(lines)

    def _fragment_dir(self, cleaned: CleanedAudio) -> Path:
        """Per-video folder, named after the metadata title or the file stem."""
        label = cleaned.metadata.title.strip() or cleaned.file.stem
        return self.output_dir / FRAGMENTS_DIRNAME / _slugify(label, max_length=80)

    def _plan(self, index: int, phrase, directory: Path, source) -> _Cut | None:
        """Resolve one phrase to sample indices, or record why it cannot be cut.

        The index is taken from the position in ``Phrases`` rather than from a
        counter over the successful cuts, so a skipped phrase leaves a gap in the
        numbering instead of silently shifting every later filename.
        """
        start_frame = max(0, round((float(phrase.start) - self.margin) * source.sample_rate))
        end_frame = min(source.frames, round((float(phrase.end) + self.margin) * source.sample_rate))

        if float(phrase.end) < float(phrase.start):
            self._skip(
                index,
                phrase,
                f"the span ends before it starts ({phrase.start:.2f}s..{phrase.end:.2f}s)",
            )
            return None

        if start_frame >= source.frames or end_frame <= start_frame:
            self._skip(
                index,
                phrase,
                f"the interval {phrase.start:.2f}s..{phrase.end:.2f}s "
                f"(with margin) falls outside the {source.duration:.2f}s of audio",
            )
            return None

        # The index prefix makes the name unique even when two phrases share
        # their text, or when their slugs collapse to the same string.
        name = f"{index:03d}_{_slugify(phrase.text)}.wav"
        return _Cut(
            index=index,
            id=phrase.id,
            text=phrase.text,
            path=directory / name,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_rate=source.sample_rate,
        )

    def _skip(self, index: int, phrase, reason: str) -> None:
        self.skipped.append(
            SkippedPhrase(index=index, id=phrase.id, text=phrase.text, reason=reason)
        )

    def _batches(self, cuts: list[_Cut]):
        for start in range(0, len(cuts), self.batch_size):
            yield cuts[start : start + self.batch_size]

    def _command(self, source: Path, batch: list[_Cut], info) -> list[str]:
        """Build the single ffmpeg invocation that writes the whole batch."""
        total = len(batch)
        # One explicit split, then one trimmed branch per output. asplit is not
        # strictly required here (ffmpeg tolerates a reused input label), but
        # stating it makes the graph's fan-out obvious instead of implicit.
        split = f"[0:a]asplit={total}" + "".join(f"[s{i}]" for i in range(total))
        branches = []
        for position, cut in enumerate(batch):
            branches.append(
                f"[s{position}]atrim=start_sample={cut.start_frame}:"
                f"end_sample={cut.end_frame},asetpts=PTS-STARTPTS[a{position}]"
            )
        graph = ";".join([split, *branches])

        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-v", "error",
            "-i", str(source),
            "-filter_complex", graph,
        ]
        for position, cut in enumerate(batch):
            # The source format is mirrored on every output: the fragment is
            # meant to be the same audio in a shorter file, not a resampled one.
            command += [
                "-map", f"[a{position}]",
                "-c:a", "pcm_s16le",
                "-ar", str(info.sample_rate),
                "-ac", str(info.channels),
                str(cut.path),
            ]
        return command

    def _run_batch(self, source: Path, batch: list[_Cut], info) -> None:
        command = self._command(source, batch, info)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise StepOutputError(
                f"ffmpeg timed out after {self.timeout}s while cutting "
                f"{len(batch)} fragments out of {source}"
            ) from exc
        if result.returncode != 0:
            details = (result.stderr or "").strip().splitlines()
            reason = details[-1] if details else f"exit code {result.returncode}"
            raise StepOutputError(
                f"ffmpeg could not cut {len(batch)} fragments out of {source}: {reason}"
            )

        for cut in batch:
            self._verify(cut)

    @staticmethod
    def _verify(cut: _Cut) -> None:
        """The file must hold exactly the samples that were asked for.

        A silent mismatch here is the one failure that would reach the Anki deck
        as a card with the wrong audio, so it is checked per fragment instead of
        inferred from ffmpeg's exit code.
        """
        info = verify_wav(cut.path)
        if info.frames != cut.frames:
            raise StepOutputError(
                f"Fragment {cut.path.name} holds {info.frames} samples but "
                f"{cut.frames} were cut "
                f"({cut.start:.3f}s..{cut.end:.3f}s); the audio would not match "
                f"its reported timings"
            )
        if info.sample_rate != cut.sample_rate:
            raise StepOutputError(
                f"Fragment {cut.path.name} is at {info.sample_rate} Hz but the "
                f"source is at {cut.sample_rate} Hz"
            )

    @staticmethod
    def _require_ffmpeg() -> None:
        if shutil.which("ffmpeg") is None:
            raise StepOutputError(
                "ffmpeg was not found in PATH; it is required to cut audio"
            )


__all__ = ["SkippedPhrase", "Step5Cut"]