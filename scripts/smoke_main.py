#!/usr/bin/env python3
"""Smoke test for the orchestrator.

The seven steps have their own smoke scripts; what is untested here is what
only exists once they are wired together: that each output reaches the next
input, that the two terminal questions behave, and that a failure stops the
run instead of handing a half-built dataset to Anki.

No network and no models. The step classes are replaced with recording stubs,
so the chain is checked for the wrong reasons-free way: the orchestrator is the
only module that knows all seven exist, and the failure this guards against is
passing ``Phrases`` where ``Fragments`` is expected — a mistake no individual
step test can catch, because each of them is called correctly in isolation.

    python scripts/smoke_main.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import wave
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.main as main_module  # noqa: E402
from pipeline.contracts import (  # noqa: E402
    AnkiPackage,
    Card,
    CleanedAudio,
    Dataset,
    DownloadedAudio,
    Fragment,
    Fragments,
    Metadata,
    Phrase,
    Phrases,
    Transcript,
    Word,
)
from pipeline.errors import StepOutputError  # noqa: E402

META = Metadata(
    url="https://www.youtube.com/watch?v=fixture",
    source="youtube",
    title="Fixture Video",
    artist="Fixture Artist",
)


def wav(path: Path, seconds: float = 0.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(seconds * 16000))
    return path


class Recorder:
    """Records which step received which argument, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []


class FakeStep1:
    def __init__(self, recorder, tmp):
        self.r, self.tmp = recorder, tmp

    def run(self, url):
        self.r.calls.append(("step1", (url,)))
        return DownloadedAudio(file=wav(self.tmp / "raw.wav"), metadata=META)


class FakeStep2:
    def __init__(self, recorder):
        self.r = recorder

    def run(self, downloaded):
        self.r.calls.append(("step2", (downloaded,)))
        assert isinstance(downloaded, DownloadedAudio), downloaded
        return CleanedAudio(file=self.r.tmp / "clean.wav", metadata=downloaded.metadata)


class FakeStep3:
    def __init__(self, recorder):
        self.r = recorder

    def run(self, cleaned):
        self.r.calls.append(("step3", (cleaned,)))
        assert isinstance(cleaned, CleanedAudio), cleaned
        return Transcript(
            text="i am twenty years old you look great today",
            language="en",
            words=[Word("i", 0.0, 0.1), Word("am", 0.1, 0.2)],
        )


class FakeStep4:
    def __init__(self, recorder):
        self.r = recorder

    def run(self, transcript, phrases):
        self.r.calls.append(("step4", (transcript, tuple(phrases))))
        assert isinstance(transcript, Transcript), transcript
        return Phrases(
            phrases=[Phrase(id="p001", text=phrases[0], start=0.0, end=0.2)]
        )

    def report(self):
        return "aligned 1/1 phrases"


class FakeStep5:
    def __init__(self, recorder):
        self.r = recorder

    def run(self, cleaned, phrases):
        self.r.calls.append(("step5", (cleaned, phrases)))
        assert isinstance(cleaned, CleanedAudio), cleaned
        assert isinstance(phrases, Phrases), phrases
        return Fragments(
            fragments=[
                Fragment(id=p.id, file=self.r.tmp / f"{p.id}.wav", start=p.start, end=p.end)
                for p in phrases.phrases
            ]
        )

    def report(self):
        return "cut 1 fragments"


class FakeStep6:
    def __init__(self, recorder, context=None, require_raises=False):
        self.r = recorder
        self.context = context
        self.require_raises = require_raises

    def require_server(self):
        self.r.calls.append(("step6.health", ()))
        if self.require_raises:
            raise StepOutputError("No llama-server answering at http://127.0.0.1:8080")

    def run(self, phrases, fragments):
        self.r.calls.append(("step6", (phrases, fragments)))
        assert isinstance(phrases, Phrases), phrases
        assert isinstance(fragments, Fragments), fragments
        return Dataset(
            cards=[
                Card(
                    id=fragment.id,
                    en=phrase.text,
                    es="Tengo 20 años",
                    audio=fragment.file,
                    start=fragment.start,
                    end=fragment.end,
                )
                for phrase, fragment in zip(phrases.phrases, fragments.fragments)
            ]
        )

    def report(self):
        return "translated 1/1 phrases"


class FakeStep7:
    def __init__(self, recorder):
        self.r = recorder

    def run(self, dataset, deck_name):
        self.r.calls.append(("step7", (dataset, deck_name)))
        assert isinstance(dataset, Dataset), dataset
        return AnkiPackage(apkg_path=self.r.tmp / f"{deck_name}.apkg")

    def report(self):
        return "exported 1 cards"


def install_stubs(recorder, tmp, require_raises=False):
    main_module.Step1Download = lambda: FakeStep1(recorder, tmp)
    main_module.Step2Denoise = lambda: FakeStep2(recorder)
    main_module.Step3Transcribe = lambda: FakeStep3(recorder)
    main_module.Step4Align = lambda: FakeStep4(recorder)
    main_module.Step5Cut = lambda: FakeStep5(recorder)
    main_module.Step6Translate = lambda context=None: FakeStep6(
        recorder, context=context, require_raises=require_raises
    )
    main_module.Step7Anki = lambda: FakeStep7(recorder)


def write_phrases(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# frases de estudio\n"
        "\n"
        "I'm twenty years old\n"
        "You look great today\n",
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------ checks


def check_chain(tmp: Path) -> None:
    print("== the outputs are wired to the next inputs ==")
    recorder = Recorder()
    recorder.tmp = tmp
    install_stubs(recorder, tmp)

    phrases = write_phrases(tmp / "frases.txt")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        dataset = main_module.run(
            "https://youtube.com/watch?v=x", phrases_path=phrases, deck_name="Mazo"
        )

    order = [name for name, _ in recorder.calls]
    assert order == [
        "step1",
        "step2",
        "step3",
        "step4",
        "step5",
        "step6.health",
        "step6",
        "step7",
    ], order
    print(f"  {' -> '.join(order)}")

    # Each step got the previous one's actual output, not a fresh fixture.
    by_name = dict(recorder.calls)
    step2_in = by_name["step2"][0]
    assert isinstance(step2_in, DownloadedAudio), step2_in
    step4_in = by_name["step4"][1]
    assert step4_in == ("I'm twenty years old", "You look great today"), step4_in
    assert isinstance(dataset, Dataset) and dataset.cards
    print("  every step received the previous step's real output")

    out = buffer.getvalue()
    for step in range(1, 8):
        assert f"paso {step}:" in out, f"missing header for step {step}\n{out}"
    assert "Mazo.apkg" in out
    print("  each step announces itself and the final path is shown")


def check_context_reaches_step6(tmp: Path) -> None:
    print("== step 6 gets the video as context ==")
    recorder = Recorder()
    recorder.tmp = tmp
    install_stubs(recorder, tmp)

    with redirect_stdout(io.StringIO()):
        main_module.run(
            "https://youtube.com/watch?v=x",
            phrases_path=write_phrases(tmp / "f2.txt"),
            deck_name="Mazo",
        )

    step6 = recorder.calls[[n for n, _ in recorder.calls].index("step6")]
    context = recorder.calls[[n for n, _ in recorder.calls].index("step6.health")]
    del step6, context
    # The context is passed to the constructor, not to run().
    assert FakeStep6.last_context is not None, "step 6 received no context"
    assert "Fixture Video" in FakeStep6.last_context, FakeStep6.last_context
    print(f"  context={FakeStep6.last_context!r}")


def check_missing_server(tmp: Path) -> None:
    print("== a missing llama-server stops the run before translating ==")
    recorder = Recorder()
    recorder.tmp = tmp
    install_stubs(recorder, tmp, require_raises=True)

    phrases = write_phrases(tmp / "f3.txt")
    try:
        with redirect_stdout(io.StringIO()):
            main_module.run("https://youtube.com/watch?v=x", phrases_path=phrases)
    except StepOutputError as exc:
        assert "llama-server" in str(exc), exc
    else:
        raise AssertionError("a missing server should have stopped the run")

    names = [name for name, _ in recorder.calls]
    # Steps 1-5 ran and their artefacts are on disk; step 6 never translated
    # and step 7 was never reached.
    assert "step6" not in names and "step7" not in names, names
    assert names[-1] == "step6.health", names
    print("  aborted after the health check, no translation and no export")


def check_failed_step_aborts(tmp: Path) -> None:
    print("== a failing step stops the run ==")
    recorder = Recorder()
    recorder.tmp = tmp
    install_stubs(recorder, tmp)

    def boom(_downloaded):
        recorder.calls.append(("step2", ()))
        raise StepOutputError("ffmpeg could not denoise the audio: exit 1")

    main_module.Step2Denoise = lambda: type(
        "Failing", (), {"run": staticmethod(boom)}
    )()

    try:
        with redirect_stdout(io.StringIO()):
            main_module.run("https://youtube.com/watch?v=x")
    except StepOutputError as exc:
        assert "ffmpeg" in str(exc), exc
    else:
        raise AssertionError("step 2 failing should have stopped the run")

    names = [name for name, _ in recorder.calls]
    assert "step3" not in names and "step7" not in names, names
    print("  later steps never ran")


def check_phrases_file(tmp: Path) -> None:
    print("== the phrases file ==")
    path = write_phrases(tmp / "notas" / "frases.txt")
    loaded = main_module._load_phrases(path)
    assert loaded == ["I'm twenty years old", "You look great today"], loaded
    print("  blank lines and # comments ignored")

    for bad, reason in [
        (tmp / "missing.txt", "does not exist"),
        (tmp / "vacio.txt", "no phrases"),
    ]:
        if reason == "no phrases":
            bad.write_text("# sólo comentarios\n\n", encoding="utf-8")
        try:
            main_module._load_phrases(bad)
        except StepOutputError as exc:
            assert reason in str(exc), exc
        else:
            raise AssertionError(f"{bad} should have been rejected")
    print("  a missing or empty file is refused")


def check_prompts(tmp: Path) -> None:
    print("== the terminal questions ==")
    import builtins

    original = builtins.input
    try:
        # An empty answer re-asks rather than returning "" and failing later.
        answers = iter(["", "  ", "https://youtube.com/watch?v=x"])
        builtins.input = lambda _prompt="": next(answers)
        assert main_module.ask_url() == "https://youtube.com/watch?v=x"
        print("  an empty URL is re-asked")

        # A closed stdin is a run that cannot ask, not a crash.
        def closed(_prompt=""):
            raise EOFError

        builtins.input = closed
        try:
            main_module.ask("anything: ")
        except StepOutputError as exc:
            assert "interactive shell" in str(exc), exc
        else:
            raise AssertionError("EOF should have been reported, not crashed")
        print("  a closed stdin fails with an explanation")
    finally:
        builtins.input = original


def check_deck_confirmation(tmp: Path) -> None:
    print("== the deck name is confirmed ==")
    import builtins

    original = builtins.input
    try:
        answers = iter(["Inglés", "s"])
        builtins.input = lambda _prompt="": next(answers)
        assert main_module.ask_deck_name() == "Inglés"
        print("  confirming keeps the name")

        answers = iter(["Inglés", "no"])
        builtins.input = lambda _prompt="": next(answers)
        try:
            main_module.ask_deck_name()
        except StepOutputError as exc:
            assert "Cancelled" in str(exc), exc
        else:
            raise AssertionError("declining should have stopped the run")
        print("  declining aborts before anything is written")
    finally:
        builtins.input = original


def check_exit_codes(tmp: Path) -> None:
    print("== main() exit codes ==")
    import builtins

    original = builtins.input
    recorder = Recorder()
    recorder.tmp = tmp
    install_stubs(recorder, tmp)
    try:
        # Ctrl-C mid-run, the way a user stops a long transcription.
        def interrupt(_prompt=""):
            raise KeyboardInterrupt

        builtins.input = interrupt
        assert main_module.main() == 130
        print("  KeyboardInterrupt -> 130")

        # A step failure: the URL is asked, then step 1 rejects it.
        answers = iter(["https://example.com/nope"])

        def give_url(_prompt=""):
            return next(answers)

        builtins.input = give_url

        def unsupported(_url):
            raise StepOutputError("Unsupported URL; only YouTube and Spotify")

        main_module.Step1Download = lambda: type(
            "Failing", (), {"run": staticmethod(unsupported)}
        )()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            assert main_module.main() == 1
        assert "Abortando" in buffer.getvalue(), buffer.getvalue()
        print("  StepOutputError -> 1, with the message and what survives")
    finally:
        builtins.input = original


FakeStep6.last_context = None
_original_init = FakeStep6.__init__


def _tracking_init(self, recorder, context=None, require_raises=False):
    FakeStep6.last_context = context
    _original_init(self, recorder, context=context, require_raises=require_raises)


FakeStep6.__init__ = _tracking_init


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        check_chain(workspace)
        check_context_reaches_step6(workspace)
        check_missing_server(workspace)
        check_failed_step_aborts(workspace)
        check_phrases_file(workspace)
        check_prompts(workspace)
        check_deck_confirmation(workspace)
        check_exit_codes(workspace)

    print("\nall good")