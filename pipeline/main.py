"""The orchestrator: ask for a URL and run the seven steps in order.

This is the only module that knows the whole chain exists. It connects each
step's output to the next step's input and nothing more: it never reaches into
a step's internals, only into the contracts in ``pipeline/contracts.py``, so
replacing any step's implementation leaves this file untouched.

Two things this deliberately does not do:

* **It does not start llama-server.** Step 6 reuses one across videos because
  loading the model costs seconds and more than a gigabyte; managing that
  process is the orchestrator's job, and the job here is to *check* it. If no
  server answers, the flow stops with the exact command to run in another
  terminal, and the artefacts written so far stay on disk.
* **It does not recover from a failed step.** A step that produces nothing
  usable raises ``StepOutputError`` and the run ends with the message. Retrying
  or skipping would leave the run in a state that is neither reviewable nor
  reproducible, and the PRD asks for the flow to stop.

Progress is printed as the chain advances, since the run is dominated by long
stretches of silence (the transcription and the translation) where it would
otherwise look hung.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline.contracts import Dataset, DownloadedAudio, Fragments, Phrases
from pipeline.errors import StepOutputError
from pipeline.steps.step1_download import Step1Download
from pipeline.steps.step2_denoise import Step2Denoise
from pipeline.steps.step3_transcribe import Step3Transcribe
from pipeline.steps.step4_align import Step4Align
from pipeline.steps.step5_cut import Step5Cut
from pipeline.steps.step6_translate import Step6Translate
from pipeline.steps.step7_anki import Step7Anki

DEFAULT_DECK_NAME = "Inglés"

# One line per phrase. Blank lines and ``#`` comments are ignored, so the file
# can carry notes for the user without breaking the run.
PHRASES_SUFFIX = ".txt"


def announce(step: int, title: str) -> None:
    print(f"== paso {step}: {title} ==", flush=True)


def report(title: str, body: str) -> None:
    """Print a step's own report, indented under it."""
    if not body:
        return
    for line in body.splitlines():
        print(f"   {line}", flush=True)
    print("", flush=True)


def ask(prompt: str) -> str:
    """Read one line from the terminal, re-asking while it comes back empty."""
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            # A closed stdin (a piped run, a CI job) is not a mistake to crash
            # on; it is a run that cannot ask questions.
            raise StepOutputError(
                "No terminal input available; run this from an interactive shell"
            ) from None
        if answer:
            return answer
        print("   (no vacío, por favor)", flush=True)


def ask_url() -> str:
    print("Pipeline de audio -> tarjetas de Anki", flush=True)
    print("", flush=True)
    return ask("URL del video (YouTube o Spotify): ")


def _resolve_phrases(raw: str, audio_dir: Path) -> Path:
    """Resolve what the user typed into an existing path.

    A relative path is tried against the working directory first, because that
    is what someone who typed it expects, and against the audio directory
    second, since that is where the file is suggested to live. Resolving only
    against the audio directory rejected the very path printed as a suggestion
    ("output/frases.txt" became "output/output/frases.txt").
    """
    path = Path(raw).expanduser()
    if path.is_absolute() or path.exists():
        return path
    beside_audio = audio_dir / path
    return beside_audio if beside_audio.exists() else path


def read_phrases(audio_dir: Path) -> list[str]:
    """Ask for the study phrases and load them from a ``.txt`` file.

    The user writes the file themselves, next to the audio, and just confirms
    the path. Keeping the list in a file rather than pasting it into the
    terminal is what makes a run reproducible: the same file with the same
    video yields the same dataset, and editing the phrases later does not
    require retyping them.
    """
    print("", flush=True)
    print("Paso 4 necesita la lista de frases de estudio.", flush=True)
    print(
        "Es un fichero .txt con una frase por línea (los comentarios con # se"
        " ignoran).",
        flush=True,
    )
    print(f"Sugerencia: créalo en {audio_dir / f'frases{PHRASES_SUFFIX}'}", flush=True)
    print("", flush=True)

    while True:
        raw = ask("Ruta del fichero de frases: ")
        path = _resolve_phrases(raw, audio_dir)
        if not path.exists():
            print(f"   no existe: {path}", flush=True)
            print("   (ruta absoluta, o relativa al directorio actual)", flush=True)
            continue
        if path.is_dir():
            print(f"   es un directorio, no un fichero: {path}", flush=True)
            continue

        phrases = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        phrases = [line for line in phrases if line and not line.startswith("#")]
        if not phrases:
            print(f"   el fichero no tiene frases: {path}", flush=True)
            continue
        return phrases


def ask_deck_name() -> str:
    print("", flush=True)
    print("Paso 7 necesita el nombre del mazo.", flush=True)
    print(
        "Al importar, Anki añade las tarjetas a un mazo con ese nombre o crea"
        " uno nuevo.",
        flush=True,
    )
    print("", flush=True)
    name = ask("Nombre del mazo: ")
    confirmation = ask(f"¿Confirmas «{name}»? [S/n] ").strip().lower()
    if confirmation.startswith("n"):
        raise StepOutputError("Cancelled at the deck name; nothing was exported")
    return name


def run(url: str, phrases_path: Path | None = None, deck_name: str | None = None) -> Dataset:
    """Run the seven steps over ``url`` and return the dataset they produced.

    ``phrases_path`` and ``deck_name`` let a caller (or a test) skip the two
    interactive questions; from a terminal both are asked for.
    """
    announce(1, "descarga")
    downloaded: DownloadedAudio = Step1Download().run(url)
    meta = downloaded.metadata
    print(f"   {meta.title} — {meta.artist}", flush=True)
    print(f"   {downloaded.file}", flush=True)

    announce(2, "limpieza de ruido")
    cleaned = Step2Denoise().run(downloaded)
    print(f"   {cleaned.file}", flush=True)

    announce(3, "transcripción (puede tardar)")
    transcript = Step3Transcribe().run(cleaned)
    print(f"   {len(transcript.words)} palabras, idioma {transcript.language}", flush=True)

    announce(4, "tiempos de frases")
    if phrases_path is None:
        phrases = read_phrases(cleaned.file.parent)
    else:
        phrases = _load_phrases(phrases_path)
    align = Step4Align()
    aligned: Phrases = align.run(transcript, phrases)
    report("paso 4", align.report())

    announce(5, "fragmentos de audio")
    cut = Step5Cut()
    fragments: Fragments = cut.run(cleaned, aligned)
    report("paso 5", cut.report())

    announce(6, "dataset (traducción; necesita llama-server)")
    translate = Step6Translate(context=_context(meta))
    # Checked before the long run rather than by the step itself, so the
    # missing-server message arrives before the user waits for a translation
    # that cannot happen.
    translate.require_server()
    dataset = translate.run(aligned, fragments)
    report("paso 6", translate.report())

    announce(7, "exportación a Anki")
    name = deck_name if deck_name is not None else ask_deck_name()
    export = Step7Anki()
    package = export.run(dataset, name)
    report("paso 7", export.report())

    print(f"Listo: {package.apkg_path}", flush=True)
    print("Importa ese fichero en Anki para empezar a estudiar.", flush=True)
    return dataset


def _load_phrases(path: Path) -> list[str]:
    """Read a phrases file given directly, with the same rules as the prompt."""
    path = Path(path)
    if not path.exists():
        raise StepOutputError(f"The phrases file does not exist: {path}")
    phrases = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    phrases = [line for line in phrases if line and not line.startswith("#")]
    if not phrases:
        raise StepOutputError(f"The phrases file has no phrases in it: {path}")
    return phrases


def _context(meta) -> str:
    """Give step 6 the video's identity, so the translations stay coherent."""
    parts = [meta.title]
    if meta.artist:
        parts.append(meta.artist)
    return " — ".join(parts)


def main() -> int:
    try:
        url = ask_url()
        run(url)
    except StepOutputError as exc:
        print("", flush=True)
        print(f"!! {exc}", flush=True)
        print(
            "Abortando. Lo ya escrito en output/ se conserva.", flush=True
        )
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())