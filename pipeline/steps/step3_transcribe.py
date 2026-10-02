"""Step 3 — transcribe the cleaned audio with per-word timestamps.

Uses ``faster-whisper`` (CTranslate2, C++) with the ``small`` English model
quantized to int8, which is the sweet spot between accuracy and CPU speed on
this machine (4 AVX2 cores, no GPU).

Two settings carry the whole step:

* ``word_timestamps=True`` produces the ``{word, start, end}`` stream that
  step 4 matches phrases against.
* ``vad_filter=True`` (Silero VAD, ONNX) drops silences and residual noise, and
  improves the temporal alignment of the words that are kept. faster-whisper
  maps the timings back onto the original timeline, so they stay comparable with
  the audio of steps 2 and 5.

Requires ``av<19``: faster-whisper 1.2.1 calls ``av.open(metadata_errors=...)``,
a keyword that PyAV 19 removed.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline.contracts import CleanedAudio, Transcript, Word
from pipeline.errors import StepOutputError
from pipeline.media import verify_wav

# The exact repo faster-whisper expects; tools-required.md §7 stores it there.
DEFAULT_MODEL_DIR = Path("models/faster-whisper-small")
FALLBACK_MODEL = "Systran/faster-whisper-small"

DEFAULT_LANGUAGE = "en"
DEFAULT_BEAM_SIZE = 5

# faster-whisper derives this from onnxruntime; pinned so the model and the VAD
# use all the cores we have instead of a single one.
DEFAULT_CPU_THREADS = os.cpu_count() or 4

# A word whose timestamps are missing or inconsistent would break step 4, and a
# word further away than this from the audio is a decoding artefact, not speech.
TIMESTAMP_TOLERANCE_SECONDS = 2.0


class Step3Transcribe:
    """Transcribes a ``CleanedAudio`` into a ``Transcript`` with word timings."""

    def __init__(
        self,
        model: str | Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = DEFAULT_LANGUAGE,
        beam_size: int = DEFAULT_BEAM_SIZE,
        cpu_threads: int = DEFAULT_CPU_THREADS,
        word_timestamps: bool = True,
        vad_filter: bool = True,
    ) -> None:
        self.model = self._resolve_model(model)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self._whisper = None

    @staticmethod
    def _resolve_model(model: str | Path | None) -> str:
        """Accept a local model directory or a Hugging Face repo id.

        Defaults to the copy in ``models/``, and falls back to the repo name so
        faster-whisper downloads it on first use if it is missing.
        """
        if model is not None:
            return str(model)
        if DEFAULT_MODEL_DIR.is_dir():
            return str(DEFAULT_MODEL_DIR)
        return FALLBACK_MODEL

    def load_model(self):
        """Instantiate the WhisperModel once; reused across videos."""
        if self._whisper is not None:
            return self._whisper
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise StepOutputError(
                "faster-whisper is not installed in this environment "
                "(uv pip install faster-whisper)"
            ) from exc

        try:
            self._whisper = WhisperModel(
                self.model,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise StepOutputError(
                f"Could not load the Whisper model {self.model!r} "
                f"(device={self.device}, compute_type={self.compute_type}): {exc}"
            ) from exc
        return self._whisper

    def run(self, cleaned: CleanedAudio) -> Transcript:
        if cleaned.is_empty():
            raise StepOutputError(
                f"Step 3 received an empty input: {cleaned.file}"
            )
        info = verify_wav(cleaned.file)
        model = self.load_model()

        try:
            segments, info_out = model.transcribe(
                str(cleaned.file),
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
            )
            segments = list(segments)
        except Exception as exc:
            raise StepOutputError(
                f"faster-whisper could not transcribe {cleaned.file}: {exc}"
            ) from exc

        text = " ".join(segment.text.strip() for segment in segments).strip()
        words = self._collect_words(segments, info.duration)
        transcript = Transcript(text=text, language=info_out.language, words=words)
        if transcript.is_empty():
            raise StepOutputError(
                f"Step 3 transcribed no speech in {cleaned.file} "
                f"({info.duration:.2f}s of audio, model {self.model})"
            )
        return transcript

    def _collect_words(self, segments, audio_duration: float) -> list[Word]:
        """Flatten the segment words into the contract, dropping unusable ones.

        faster-whisper may leave ``start``/``end`` unset on a word it could not
        align, and the very last word can drift past the end of the file. Those
        would silently shift the phrase timings of step 4, so they are discarded
        here instead of being passed along.
        """
        limit = audio_duration + TIMESTAMP_TOLERANCE_SECONDS
        words: list[Word] = []
        for segment in segments:
            for word in segment.words or ():
                start, end = word.start, word.end
                if start is None or end is None:
                    continue
                if start < 0 or end < start or start > limit:
                    continue
                text = word.word.strip()
                if not text:
                    continue
                words.append(Word(word=text, start=round(start, 3), end=round(end, 3)))
        return words
