# PRD — Pipeline de Procesamiento de Audio

## 1. Visión general

Proyecto que ejecuta varios pasos hasta obtener un resultado final. Cada paso se implementará con el lenguaje/librería más eficiente para esa tarea (se permite múltiples lenguajes). Este PRD documenta el flujo completo en 7 pasos: a partir del URL de un video se genera un paquete de tarjetas para Anki con frases en inglés, su traducción idiomática al español y el audio de cada frase.

## 2. Arquitectura y desacoplamiento

- Un solo proyecto separado en **módulos por paso** (carpeta `pipeline/steps/`), cada uno independiente y reemplazable.
- Cada módulo expone una clase con método `run(input) -> output`.
- **El contrato es la dataclass de salida** (definidas en `pipeline/contracts.py`): la salida de cada paso es la entrada del siguiente y es lo único estable. La implementación interna de cualquier paso puede cambiarse (otra librería, otro motor, incluso otro lenguaje) mientras conserve su contrato.
- Los pasos no se acoplan entre sí: ninguno llama métodos internos de otro ni conoce su implementación; solo usan los tipos del contrato.
- El **consumidor final no acopla a los pasos intermedios**: Anki es únicamente un consumidor de la salida del paso 7 y puede reemplazarse por una app propia sin tocar los pasos 1–6.
- **Orquestador** (`pipeline/main.py`): al ejecutarse en terminal solicita la URL y ejecuta la cadena conectando la salida de cada paso a la entrada del siguiente.
- **Manejo de errores:** si un paso no produce salida (None, vacío o error), se lanza una excepción `StepOutputError` y se detiene el flujo.

### Estructura del proyecto

```
video/
  PRD.md
  pipeline/
    main.py            # orquestador: pide URL, encadena y maneja excepciones
    contracts.py       # dataclasses estables (contratos entre pasos)
    steps/
      step1_download.py
      step2_denoise.py
      step3_transcribe.py
      step4_align.py
      step5_cut.py
      step6_translate.py
      step7_anki.py
```

### Contratos entre pasos

| # | Paso | Entrada | Salida (contrato) |
|---|---|---|---|
| 1 | Descarga | URL | `DownloadedAudio(file, metadata)` |
| 2 | Denoising | `DownloadedAudio` | `CleanedAudio(file, metadata)` |
| 3 | Transcripción | `CleanedAudio` | `Transcript(text, language, words[Word])` |
| 4 | Tiempos de frases | `Transcript` + frases | `Phrases(phrases[Phrase])` |
| 5 | Fragmentos | `CleanedAudio` + `Phrases` | `Fragments(fragments[Fragment])` |
| 6 | Dataset | `Phrases` + `Fragments` | `Dataset(cards[Card])` |
| 7 | Exportar Anki | `Dataset` | `AnkiPackage(apkg_path)` |

Detalle de los contratos:

- `Word{word, start, end}`
- `Phrase{id, text, start, end}`
- `Fragment{id, file, start, end}`
- `Card{id, en, es, audio, start, end}` → información organizada del video, **agnóstica al consumidor**.
- `Metadata{url, source, title, artist}` → fluye a lo largo del pipeline sin acoplar los pasos.

## 3. Paso 1: Descarga de audio

- **Entrada:** URL de YouTube o de Spotify.
- **Salida (`DownloadedAudio`):** archivo de audio local en **`.wav` (PCM)** como formato intermedio sin pérdida + metadatos (título, artista, álbum, etc.).
- **Requisito clave:** poder acceder al nombre del audio.

### Solución (Paso 1)

- **Lenguaje:** Python (la herramienta de facto es Python para ambas plataformas).
- **YouTube:** librería `yt-dlp`. Se extrae el `info` dict (título, artista, uploader) y se usa `bestaudio/best` + postprocesador `FFmpegExtractAudio` para convertir a WAV.
- **Spotify:** `spotdl`, que devuelve metadatos (`title`, `artists`, `album_name`) y descarga buscando el audio en YouTube (integra ambas plataformas).
- **Dependencia externa:** FFmpeg (conversión/extracción).
- **Módulo:** `step1_download` → `run(url) -> DownloadedAudio`.
- **Nota de eficiencia:** el cuello de botella es red + FFmpeg, no Python; no hay pérdida real frente a Rust/Go/C++ para este paso, y ninguna librería de esos lenguajes alcanza la paridad de características.
- **Decisión de formato:** el WAV (PCM) evita pérdidas entre pasos y re-encodeos. La conversión a formato "publicable" (mp3/m4a/opus) se difiere a un paso/consumidor final.

### Criterios de éxito (Paso 1)

1. Descargar audio a partir de una URL de YouTube y una de Spotify.
2. Retornar el nombre/título del audio.
3. Archivo de salida en formato WAV con metadatos accesibles.

## 4. Paso 2: Eliminación de ruido de fondo

- **Entrada (`DownloadedAudio`):** archivo `.wav` del paso 1.
- **Salida (`CleanedAudio`):** audio limpio en `.wav` + metadatos preservados.
- **Contexto:** predominancia de voz; prioridad de calidad buena con tiempo de procesamiento adecuado (ni demasiado rápido ni lento).

### Solución (Paso 2)

- **Base:** C vía FFmpeg.
- **Filtro principal:** `arnndn` (RNNoise) — state-of-the-art para supresión de ruido en voz, baja CPU, procesa WAV directo (internamente re-muestrea a 48 kHz y regresa).
- **Módulo:** `step2_denoise` → `run(DownloadedAudio) -> CleanedAudio`.
- **Alternativas documentadas:**
  - `afftdn` (FFmpeg): denoising espectral automático, más rápido aún, resultado aceptable.
  - `noisereduce` (Python): spectral gating, mayor control/calidad en casos complejos, más lento (numpy) y suele requerir una sección de "solo ruido" para aprender el perfil.
- **Nota de eficiencia:** el procesamiento en C (FFmpeg) es eficiente y, al operar sobre WAV sin re-encodear, no introduce pérdidas ni conversiones intermedias.

### Criterios de éxito (Paso 2)

1. Reducir el ruido de fondo de una muestra de voz manteniendo la inteligibilidad.
2. Sin degradación apreciable del audio (voz natural, sin artefactos).
3. Preservar los metadatos del audio original.
4. Tiempo de procesamiento adecuado sin dominar el pipeline.

## 5. Paso 3: Transcripción con timestamps por palabra

- **Entrada (`CleanedAudio`):** archivo `.wav` limpio del paso 2.
- **Salida (`Transcript`):** texto transcrito + secuencia de palabras con tiempos `{word, start, end}` + idioma.
- **Contexto:** audio en inglés; inferencia solo en CPU; prioridad de precisión.

### Solución (Paso 3)

- **Lenguaje:** Python (`faster-whisper`), motor CTranslate2 en C++ → uso eficiente de CPU con cuantización int8.
- **Modelo:** `small` (inglés) — buena precisión balanceada para CPU.
- **Ajustes clave:**
  - `word_timestamps=True` → tiempos `start`/`end` por palabra.
  - `vad_filter=True` → descarta silencios/ruido residual y mejora la alineación temporal.
  - `compute_type` int8 (CPU).
- **Módulo:** `step3_transcribe` → `run(CleanedAudio) -> Transcript`.
- **Alternativas documentadas:**
  - `whisper.cpp` (C/C++): footprint mínimo, binario nativo con modelos ggml cuantizados; también expone timestamps por palabra (mayor ajuste).
  - `Vosk` (Python/Java/C++): streaming en tiempo real con modelos de ~40 MB y `start`/`end` por palabra; el más ligero, sin GPU.

### Criterios de éxito (Paso 3)

1. Transcripción precisa del audio en inglés.
2. Timestamps por palabra coherentes y alineados con el audio.
3. Tiempo de inferencia razonable en CPU (el modelo `small` debe procesar en tiempo cuasi-real o mejor).

## 6. Paso 4: Asignación de tiempos a frases

- **Entrada (`Transcript` + frases):** secuencia de palabras con tiempos + lista de frases en texto.
- **Salida (`Phrases`):** por frase, tiempos `{start, end}` (primera ocurrencia) + texto.
- **Contexto:** frases verbatim / casi exactas respecto al audio; solo se necesita la primera ocurrencia de cada frase.

### Solución (Paso 4)

- **Lenguaje:** Python.
- **Enfoque principal:** matching por tokens normalizados. Normalizar frase y transcripción (minúsculas, sin puntuación) y buscar la frase como subsecuencia de tokens en la lista de palabras del paso 3.
  - `start` = timestamp de la primera palabra de la coincidencia.
  - `end` = timestamp final de la última palabra de la coincidencia.
  - Primera ocurrencia por defecto; configurable para devolver todas las apariciones en el futuro.
- **Complejidad:** O(n·m) con recorrido lineal sobre la lista de palabras.
- **Optimización documentada:** si la lista de frases es muy grande (~miles), construir un trie (Aho-Corasick) para O(n + coincidencias).
- **Respaldo fuzzy:** si un caso "casi exacto" falla (p. ej. contracciones), usar `RapidFuzz` (C++ en Python) sobre el mismo matching de tokens.
- **Módulo:** `step4_align` → `run(Transcript, phrases) -> Phrases`.
- **Sin re-procesamiento de audio:** la alineación reutiliza la salida del paso 3; no se requiere re-inferencia.

### Criterios de éxito (Paso 4)

1. Asignar tiempos `{start, end}` a cada frase dentro de los límites temporales de sus palabras.
2. Resolver correctamente la primera ocurrencia de cada frase.
3. Detectar y reportar frases que no se encuentren en la transcripción.

## 7. Paso 5: Generación de fragmentos de audio por frase

- **Entrada (`CleanedAudio` + `Phrases`):** audio limpio del paso 2 + frases con tiempos del paso 4.
- **Salida (`Fragments`):** un archivo `.wav` por frase, con margen de ~100 ms antes y después del intervalo; cortes independientes (se permite solapamiento).
- **Contexto:** volumen típico de ~100 frases por video.

### Solución (Paso 5)

- **Base:** C vía FFmpeg.
- **Enfoque principal:** una sola invocación de FFmpeg con múltiples salidas, `-ss`/`-to` por fragmento, salida WAV PCM (sin re-encodeo con pérdida). Escala bien para ~100 cortes sin overhead de un proceso por frase.
- **Margen:** `start' = max(0, start − 0.1)`, `end' = min(duración, end + 0.1)`.
- **Módulo:** `step5_cut` → `run(CleanedAudio, Phrases) -> Fragments`.
- **Nombrado:** archivo por frase con nombre legible derivado del id/texto de la frase (sanitizado).
- **Alternativa documentada:** rebanado PCM directo en Python (`wave`/`numpy`) calculando `index = round(segundos × sample_rate)`, byte-exacto, si el número de cortes crece mucho.

### Criterios de éxito (Paso 5)

1. Un archivo `.wav` por frase, con duración ≈ duración de la frase + 200 ms de margen.
2. Cortes precisos a nivel de muestra (sin corrimientos temporales).
3. Fragmentos WAV sin pérdida, con sus tiempos `{start, end}` reportados en el contrato.

## 8. Paso 6: Consolidación del dataset

- **Entrada (`Phrases` + `Fragments`):** frases en inglés con tiempos + fragmentos de audio del paso 5.
- **Salida (`Dataset`):** información organizada del video, **agnóstica al consumidor**: por frase, `{id, en, es, audio, start, end}`. No está orientada a Anki; cualquier app futura la consume igual.
- **Contexto:** ~100 frases por video; recursos: ~2 GB RAM, solo CPU; prioridad de naturalidad idiomática.

### Solución (Paso 6)

- **Base:** `llama.cpp` / `llama-cpp-python` (backend en C++).
- **Modelo:** `Llama-3.2-1B-Instruct` (alternativa: `Qwen2.5-1.5B-Instruct`) en GGUF int4 (~1 GB), CPU.
- **Enfoque:** traducción en lote de las frases de cada video en un solo prompt, con instrucción explícita de traducción equivalente/coloquial (p. ej. "I'm 20 years old" → "Tengo 20 años") y contexto del video para coherencia.
- **Módulo:** `step6_translate` → `run(Phrases, Fragments, context) -> Dataset` (une cada frase con su audio y su traducción).
- **Nota de eficiencia:** lote único evita el overhead de N inferencias; el backend C++ se integra desde el pipeline Python sin procesos externos.

### Criterios de éxito (Paso 6)

1. Traducciones naturalmente idiomáticas, sin calcos literales ("Tengo 20 años", no "Yo soy 20 años viejo").
2. Coherencia de términos dentro de un mismo video (contexto compartido).
3. Procesar ~100 frases por video en tiempo razonable con ~2 GB RAM en CPU.
4. El `Dataset` contiene la información organizada (audio + EN + ES + tiempos) sin referencia al consumidor final.

## 9. Paso 7: Exportación a Anki

- **Entrada (`Dataset`):** frases en inglés, traducciones idiomáticas en español y fragmentos de audio por frase.
- **Salida (`AnkiPackage`):** archivo `.apkg` listo para importar en Anki (todas las tarjetas de una vez, sin subir frase por frase).
- **Contexto:** estudio personal de inglés; importación offline; sin deduplicación (cada ejecución genera las notas de nuevo).
- **Decoplado:** este paso es el **único punto acoplado a Anki** y es reemplazable por cualquier otro exportador (app propia, CSV, web) sin tocar los pasos 1–6.

### Solución (Paso 7)

- **Lenguaje:** Python (`genanki`).
- **Mazo:** el script pregunta el **nombre del mazo**. Al importar el `.apkg`, Anki agrega las tarjetas a un mazo existente con ese nombre o crea uno nuevo automáticamente. Se pide confirmación del nombre antes de generar el paquete.
- **Modelo dedicado fijo** `ES-EN Audio` (id estable para reutilizarlo entre importaciones) con campos **Español**, **Inglés**, **Audio**:
  - *Frente:* `{{Español}}` (el texto en español es lo primero que se muestra).
  - *Reverso:* `{{Inglés}}` + `{{Audio}}` (el campo `Audio` con `[sound:...]` muestra el botón de reproducir de Anki).
- **Audios embebidos:** los fragmentos `.wav` van en `media_files` del paquete y se referencian como `[sound:<nombre>.wav]`.
- **Módulo:** `step7_anki` → `run(Dataset, deck_name) -> AnkiPackage`.
- **Nota:** la importación es única; no se reutiliza conexión con Anki ni add-ons.

### Criterios de éxito (Paso 7)

1. Un archivo `.apkg` que, al importarse, crea una tarjeta por frase con **frente = español** y **reverso = inglés + botón de audio**.
2. El audio está embebido en el paquete y se reproduce desde la tarjeta.
3. El usuario solo indica el nombre del mazo (existente o nuevo) y realiza una importación única.
4. Sin pasos manuales por frase.

## 10. Flujo completo del pipeline

1. **Descarga** (`step1_download`, yt-dlp / spotdl) → `DownloadedAudio` (`raw.wav` + metadatos).
2. **Denoising** (`step2_denoise`, FFmpeg/RNNoise) → `CleanedAudio`.
3. **Transcripción** (`step3_transcribe`, faster-whisper `small`) → `Transcript` (palabras con `{start, end}`).
4. **Tiempos de frases** (`step4_align`, matching por tokens) → `Phrases` (`{start, end}` por frase).
5. **Fragmentos** (`step5_cut`, FFmpeg) → `Fragments` (`.wav` por frase con margen).
6. **Dataset** (`step6_translate`, llama.cpp LLM 1–2B) → `Dataset` (información organizada: audio + EN + ES).
7. **Exportar Anki** (`step7_anki`, genanki) → `AnkiPackage` (`.apkg` con frente ES / reverso EN + audio).

## 11. Decisiones pendientes

- Formato de publicación final del audio (mp3, m4a, opus) si se requiere una salida distinta al WAV intermedio.
- Nombrado definitivo de los fragmentos de audio y del `.apkg` (por video/título).
- Mecanismo de persistencia intermedia (artefactos en disco) sin acoplar los contratos de memoria.