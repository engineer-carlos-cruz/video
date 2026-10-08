"""Step 2 — remove background noise from the downloaded audio.

Runs the FFmpeg ``arnndn`` filter (RNNoise, C) over the wav produced by step 1.
The filter is kept lossless end to end: it operates on 48 kHz PCM internally and
we write the result back as 16-bit PCM wav.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.contracts import CleanedAudio, DownloadedAudio
from pipeline.errors import StepOutputError
from pipeline.media import verify_wav

MODEL_DIR = Path("models/arnndn")

# FFmpeg's arnndn filter requires an explicit model and no longer ships a default
# one, so the RNNoise weights are fetched from the community mirror below.
# "std" is the original Xiph RNNoise model; it measured best on our fixture
# (39.0 dB SNR gain, keeping 95% of the speech level).
MODELS = {
    "std": {
        "url": "https://raw.githubusercontent.com/richardpl/arnndn-models/master/std.rnnn",
        "sha256": "6b8943dc4a9b6b24425873992a44f29c0577503276456af46a8854774faeb294",
    },
}

OUTPUT_SUFFIX = "_clean"
TARGET_SAMPLE_RATE = 48000
MAX_DURATION_DRIFT_SECONDS = 1.0 / TARGET_SAMPLE_RATE  # one sample


class Step2Denoise:
    """Denoises the audio of a ``DownloadedAudio`` into a ``CleanedAudio``."""

    def __init__(
        self,
        output_dir: Path = Path("output"),
        model: str = "std",
        mix: float = 1.0,
        timeout: float | None = None,
    ) -> None:
        if model not in MODELS:
            raise StepOutputError(
                f"Unknown arnndn model {model!r}; available: {', '.join(MODELS)}"
            )
        self.output_dir = Path(output_dir)
        self.model = model
        self.mix = mix
        self.timeout = timeout
        self._model_path: Path | None = None

    def ensure_model(self) -> Path:
        """Return the model path, downloading it once if it is not there."""
        if self._model_path is not None:
            return self._model_path

        spec = MODELS[self.model]
        path = MODEL_DIR / f"{self.model}.rnnn"
        if not self._is_valid(path, spec["sha256"]):
            self._download(path, spec["url"], spec["sha256"])
        self._model_path = path
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _is_valid(self, path: Path, expected_sha: str) -> bool:
        return path.exists() and path.stat().st_size > 0 and self._sha256(path) == expected_sha

    def _download(self, path: Path, url: str, expected_sha: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                temporary.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            raise StepOutputError(
                f"Could not download the arnndn model from {url}: {exc}. "
                f"Fetch it manually into {path}."
            ) from exc

        actual = self._sha256(temporary)
        if actual != expected_sha:
            temporary.unlink(missing_ok=True)
            raise StepOutputError(
                f"{url} returned a file with an unexpected checksum "
                f"({actual}, expected {expected_sha}); refusing to use it."
            )
        temporary.replace(path)

    def _require_ffmpeg(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise StepOutputError(
                "ffmpeg was not found in PATH; it is required to denoise audio"
            )

    def run(self, downloaded: DownloadedAudio) -> CleanedAudio:
        if downloaded.is_empty():
            raise StepOutputError(
                f"Step 2 received an empty input: {downloaded.file}"
            )
        self._require_ffmpeg()
        model_path = self.ensure_model()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"{downloaded.file.stem}{OUTPUT_SUFFIX}.wav"

        before = verify_wav(downloaded.file)

        # arnndn processes 480-sample frames (10 ms at 48 kHz) and flushes the
        # trailing partial frame, padding the output by up to 479 samples.
        # atrim restores the exact original length so the audio stays aligned
        # with the transcript timings.
        #
        # The bound is counted in the *output* stream's samples, which is
        # TARGET_SAMPLE_RATE, not in the input's frame count. Step 1 does not
        # fix a sample rate (FFmpegExtractAudio keeps the source's, and 44.1 kHz
        # is the usual one for YouTube and Spotify), so using before.frames
        # directly truncated the audio to frames/48000 seconds: a 3 s track at
        # 44.1 kHz came out 2.756 s. Measured: 16 kHz and 22.05 kHz inputs lost
        # even more. Converting the duration is what makes this hold for any
        # input rate.
        end_sample = round(before.duration * TARGET_SAMPLE_RATE)
        filter_chain = (
            f"arnndn=model={model_path.resolve()}:mix={self.mix},"
            f"atrim=end_sample={end_sample},asetpts=PTS-STARTPTS"
        )
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-v", "error",
            "-i", str(downloaded.file),
            "-af", filter_chain,
            "-c:a", "pcm_s16le",
            "-ar", str(TARGET_SAMPLE_RATE),
            str(output),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise StepOutputError(
                f"ffmpeg timed out after {self.timeout}s while denoising {downloaded.file}"
            ) from exc
        if result.returncode != 0:
            details = (result.stderr or "").strip().splitlines()
            reason = details[-1] if details else f"exit code {result.returncode}"
            raise StepOutputError(f"ffmpeg could not denoise {downloaded.file}: {reason}")

        self._verify_output(downloaded, output)
        return CleanedAudio(file=output, metadata=downloaded.metadata)

    @staticmethod
    def _verify_output(source: DownloadedAudio, output: Path) -> None:
        """The cleaned audio must stay sample-aligned with its input.

        A drift here would silently break the word timings of step 4, so it is
        checked rather than assumed.
        """
        before = verify_wav(source.file)
        after = verify_wav(output)
        drift = abs(after.duration - before.duration)
        if drift > MAX_DURATION_DRIFT_SECONDS:
            raise StepOutputError(
                f"Denoising changed the duration by {drift:.4f}s "
                f"({before.duration:.3f}s -> {after.duration:.3f}s); "
                "the audio would no longer be aligned with the transcript"
            )