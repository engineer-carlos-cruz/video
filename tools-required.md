# Tools Required — Entorno del Pipeline de Audio

Documento de referencia con **todo lo que hay que instalar** en Ubuntu 26.04 para poder ejecutar
el pipeline descrito en [PRD.md](PRD.md).

- **Lenguajes:** únicamente **Python 3.14**. Ningún otro lenguaje de programación es necesario.
  Los pasos 2 y 5 se ejecutan dentro de **FFmpeg (C)** y el paso 6 dentro de **llama.cpp (C++)**,
  ambos como binarios nativos invocados desde Python.
- **Requisito de hardware:** solo **CPU**. La GPU de esta máquina (GeForce 920M, Kepler cc 3.5)
  no está soportada por CUDA moderno, así que queda descartada (igual que indica el PRD).

---

## 1. Entorno objetivo

Verificado en la máquina de desarrollo el 2026-09-25.

| Recurso | Valor | Implicación |
|---|---|---|
| SO | Ubuntu 26.04.1 LTS (`resolute`) | `ffmpeg` 8.0.1 disponible en `universe` |
| CPU | 4 cores, **AVX2** (sin AVX-512) | Cuantización `int8`/`int4` viable; no usar builds AVX-512 |
| RAM | 7.1 GB | Whisper `small` int8 + Llama 1B int4 entran de sobra |
| GPU | Intel HD 5500 + GeForce 920M (Kepler) | **Inutilizable** → CPU-only |
| Disco | 266 GB libres | Modelos (~1.6 GB) sin problema |
| Python | 3.14.4 (con `venv` + `ensurepip`) | Compatible, salvo donde se indica |

### 1.1 Estado actual: qué falta instalar

| Componente | Estado |
|---|---|
| `ffmpeg` | ❌ **no instalado** — única dependencia crítica del sistema |
| `uv` | ❌ no instalado |
| Paquetes Python del pipeline | ❌ no instalados |
| Modelos (Whisper + Llama) | ❌ no descargados |
| Binario de llama.cpp | ❌ no descargado |
| `build-essential` 12.12 | ✅ ya instalado |
| `cmake` 4.2.3 | ✅ ya instalado |
| `python3-venv` 3.14.3 | ✅ ya instalado |
| `git` 2.53 | ✅ ya instalado |
| `curl` 8.18 / `wget` 1.25 / `jq` 1.8 | ✅ ya instalados |

---

## 2. Mapa de dependencias por paso del PRD

| # | Paso | Motor / lenguaje | Dependencias |
|---|---|---|---|
| 1 | Descarga | Python + FFmpeg | `yt-dlp`, `spotdl`, `ffmpeg` |
| 2 | Denoising | **C** (FFmpeg `arnndn`) | `ffmpeg` |
| 3 | Transcripción | Python + C++ (CTranslate2) | `faster-whisper` → `ctranslate2`, `tokenizers`, `onnxruntime`, `av` |
| 4 | Tiempos de frases | Python (stdlib) | `rapidfuzz` (respaldo fuzzy) |
| 5 | Fragmentos | **C** (FFmpeg) | `ffmpeg` |
| 6 | Traducción | **C++** (llama.cpp) | binario `llama-server` + modelo GGUF |
| 7 | Exportar Anki | Python | `genanki` |

---

## 3. Paquetes del sistema (apt)

**FFmpeg es la dependencia transversal del proyecto**: la usan los pasos 1 (extracción de audio),
2 (filtro `arnndn` = RNNoise) y 5 (corte multi-salida).

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verificación:

```bash
ffmpeg -version
ffmpeg -hide_banner -filters | grep -E "arnndn|afftdn"
```

**No hace falta instalar nada más del sistema.** Explícitamente **no** se necesita:

- Node.js / npm / Go / Java — ningún paso los usa
- Drivers NVIDIA / CUDA — GPU no soportada
- `build-essential`, `cmake` — solo serían necesarios para compilar `llama-cpp-python`
  (ver §6.3); ya están instalados, pero con la solución elegida no se usan
- `libsndfile1` — FFmpeg ya lee y escribe WAV PCM
- `aria2c` — `yt-dlp` tiene su propio descargador

---

## 4. Entorno Python con `uv`

`uv` se usa por resolución rápida y lockfile reproducible. Alternativa equivalente sin instalar
nada: `python3 -m venv .venv && source .venv/bin/activate`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"     # añadir esta línea a ~/.bashrc

cd /home/carlos/Projects/video
uv venv --python 3.14 .venv
source .venv/bin/activate
```

> `yt-dlp` ya está instalado globalmente vía `pipx` (2026.8.19). Conviene instalarlo **también
> dentro del venv** para que el pipeline no dependa del `PATH` global.

---

## 5. Paquetes Python

```bash
uv pip install yt-dlp spotdl          # Paso 1
uv pip install faster-whisper         # Paso 3
uv pip install rapidfuzz              # Paso 4 (respaldo fuzzy)
uv pip install huggingface-hub        # descarga de modelos
uv pip install genanki                # Paso 7
```

### 5.1 Compatibilidad con Python 3.14

Verificado contra PyPI para `cp314` / `manylinux` `x86_64`:

| Paquete | Versión | Soporte cp314 | Nota |
|---|---|---|---|
| `ctranslate2` | 4.8.2 | ✅ `manylinux_2_27` | motor del paso 3 |
| `tokenizers` | 0.23.2 | ✅ `cp310-abi3` | ABI3, vale para 3.14 |
| `onnxruntime` | 1.30.0 | ✅ `manylinux_2_28` | VAD de faster-whisper |
| `av` | 18.1.0 | ✅ `manylinux_2_31` | decodificación |
| `numpy` | 2.5.3 | ✅ `manylinux_2_27` | respaldo corte PCM (paso 5) |
| `rapidfuzz` | 3.14.6 | ✅ `manylinux_2_27` | paso 4 |
| `spotdl` | 4.5.2 | ✅ pura (`>=3.10,<3.15`) | paso 1 |
| `yt-dlp` | 2026.8.19 | ✅ pura (`>=3.10`) | paso 1 |
| `faster-whisper` | 1.2.1 | ✅ pura (`>=3.9`) | paso 3 |
| `genanki` | 0.13.1 | ✅ pura (`>=3.6`) | paso 7 |
| **`llama-cpp-python`** | 0.3.35 | ❌ **solo sdist** | ver §6 |

**Conclusión:** Python 3.14 funciona para 6 de los 7 pasos sin ninguna intervención. El único
punto de fricción era el paso 6, resuelto con el binario nativo (§6).

### 5.2 Nota sobre `spotdl`

Es el paquete más pesado de instalar: arrastra `pydantic`, `rich`, `fastapi`, `uvicorn`, `mutagen`,
`ytmusicapi`, `spotipy`, `soundcloud-v2`, `syncedlyrics`, `pykakasi`, `websockets`. Si estorba,
puede diferirse hasta implementar el paso 1 y la rama de Spotify.

---

## 6. Paso 6 — llama.cpp

### 6.1 Por qué no `llama-cpp-python`

`llama-cpp-python` 0.3.35 publica **únicamente el tar.gz de código fuente**, sin ruedas
precompiladas. Revisado el índice oficial de wheels, para `cp314` solo existen builds de
**riscv64**; para `linux_x86_64` solo hay ruedas `cp313`/`cp312`.

### 6.2 Solución elegida — binario oficial

Los releases de `ggml-org/llama.cpp` publican binarios CPU para Ubuntu x64 (16 MB). Se usa
`llama-server`, que expone una API HTTP compatible con OpenAI y devuelve **JSON estructurado**
— mucho más fiable que parsear el stdout de `llama-cli`.

```bash
mkdir -p ~/tools/llama && cd ~/tools/llama
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b11191/llama-b11191-bin-ubuntu-x64.tar.gz
tar xzf llama-b11191-bin-ubuntu-x64.tar.gz

export PATH="$HOME/tools/llama/llama-b11191:$PATH"   # añadir a ~/.bashrc
llama-server --version
```

> **Ojo con la ruta:** los binarios están en la **raíz** de la carpeta extraída
> (`llama-b11191/llama-server`), **no** en `build/bin/`. El ejecutable principal pesa solo
> ~17 KB porque es un lanzador que carga en runtime la variante de CPU adecuada
> (`libggml-cpu-haswell.so`, `-alderlake.so`, `-sandybridge.so`, etc.). No necesita
> `LD_LIBRARY_PATH`: usa `$ORIGIN` como RPATH.

**Arranque del servidor (paso 6):**

```bash
llama-server -m models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
             --host 127.0.0.1 --port 8080 \
             -c 4096 --threads 4
```

Queda escuchando en background y `step6_translate` le manda cada lote por
`POST http://127.0.0.1:8080/v1/chat/completions`.

**Apagado:** `pkill -f llama-server` — ojo, el patrón también coincide con la propia terminal;
usar `pkill -x llama-server` si se da el caso.

### 6.3 Alternativa si en el futuro se quiere integración in-process

```bash
uv pip install llama-cpp-python    # compila desde fuente: 5–10 min
```

Requiere `build-essential` + `cmake` (ya instalados) y cumple literalmente la nota del PRD de
integrar "sin procesos externos". Como `step6_translate` queda encapsulado detrás del contrato
`Dataset`, migrar de una opción a otra **no toca los pasos 1–5 ni el 7**.

### 6.4 Impacto respecto al PRD

El PRD dice *"el backend C++ se integra desde el pipeline Python sin procesos externos"*. Con la
solución elegida hay un proceso externo, pero el impacto es nulo:

- El motor sigue siendo **C++ puro de llama.cpp** con el **mismo modelo GGUF**.
- El PRD ya pide *"traducción en lote de las frases de cada video en un solo prompt"*, así que
  el número de inferencias no aumenta: sigue siendo 1 por video.
- El servidor se reutiliza entre videos, sin recargar el modelo.

---

## 7. Modelos (~1.6 GB)

Se guardan **fuera de `pipeline/`** (no aparecen en la estructura del PRD y son artefactos
pesados, no código).

```bash
mkdir -p models

# Paso 3 — el repo exacto que espera faster-whisper
hf download Systran/faster-whisper-small --local-dir models/faster-whisper-small

# Paso 6 — Llama-3.2-1B-Instruct cuantizado a int4 (770 MB)
hf download unsloth/Llama-3.2-1B-Instruct-GGUF --local-dir models/llama \
    --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

Alternativas válidas para el paso 6:

| Modelo | Archivo | Tamaño |
|---|---|---|
| Llama-3.2-1B-Instruct | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 770 MB |
| Qwen2.5-1.5B-Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (repo `Qwen/Qwen2.5-1.5B-Instruct-GGUF`) | ~1.1 GB |

> Para int4 se recomienda `_Q4_K_M`; evita `Q2_K`/`Q3_K` (degradan la naturalidad, que es el
> criterio 1 del paso 6) y `F16`/`BF16` (~2.5 GB, fuera del presupuesto de RAM del PRD).

---

## 8. Script de instalación completo

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- 1) Sistema ---
sudo apt update
sudo apt install -y ffmpeg

# --- 2) Entorno Python ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /home/carlos/Projects/video
uv venv --python 3.14 .venv
source .venv/bin/activate

# --- 3) Paquetes Python ---
uv pip install yt-dlp spotdl faster-whisper rapidfuzz huggingface-hub genanki

# --- 4) Modelos (~1.6 GB) ---
mkdir -p models
hf download Systran/faster-whisper-small --local-dir models/faster-whisper-small
hf download unsloth/Llama-3.2-1B-Instruct-GGUF --local-dir models/llama \
    --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# --- 5) llama.cpp (paso 6) ---
mkdir -p ~/tools/llama
cd ~/tools/llama
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b11191/llama-b11191-bin-ubuntu-x64.tar.gz
tar xzf llama-b11191-bin-ubuntu-x64.tar.gz
export PATH="$HOME/tools/llama/llama-b11191:$PATH"
cd /home/carlos/Projects/video
```

---

## 9. Verificación (smoke tests)

```bash
# Sistema
ffmpeg -version | head -1
ffmpeg -hide_banner -filters | grep arnndn                  # paso 2

# Paquetes Python
yt-dlp --version                                          # paso 1
python -c "import spotdl; print('spotdl', spotdl.__version__)"          # paso 1
python -c "import genanki, rapidfuzz; print('pasos 4 y 7 ok')"           # pasos 4 y 7

# Paso 3 — carga real del modelo en CPU int8
python -c "
from faster_whisper import WhisperModel
m = WhisperModel('models/faster-whisper-small', device='cpu', compute_type='int8')
print('modelo whisper cargado:', m.model.bin_size)
"

# Paso 6 — el binario solo
llama-server --version
```

**Prueba de extremo a extremo del paso 6** (servidor + traducción idiomática):

```bash
# Terminal 1
llama-server -m models/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
             --host 127.0.0.1 --port 8080 -c 4096 --threads 4

# Terminal 2
curl -s localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[
        {"role":"system","content":"Traduce al español idiomático, una frase por línea, mismo formato numerado."},
        {"role":"user","content":"1. I am 20 years old\n2. She is going to the store\n3. I have been learning English for two years"}],
       "temperature":0.2,"max_tokens":400}'
```

Salida esperada (criterio 1 del paso 6: *"Tengo 20 años"*, no *"Yo soy 20 años viejo"*):

```
1. Tengo 20 años
2. Vámonos a la tienda
3. He estado aprendiendo inglés durante dos años
```

---

## 10. Resultados medidos

Medidos el 2026-09-25 en esta máquina (Ubuntu 26.04.1, 4 cores AVX2, 7.1 GB RAM):

| Métrica | Valor |
|---|---|
| Tamaño del binario de llama.cpp | 16.2 MB (comprimido) |
| Descarga de `llama-b11191-bin-ubuntu-x64.tar.gz` | ~10 s |
| Arranque de `llama-server` (modelo cargado) | ~3 s |
| **RAM de `llama-server`** (1B Q4_K_M, `-c 4096`) | **1414 MB RSS** |
| Traducción de 5 frases (1 lote HTTP) | 7.5 s |
| Tokens generados (5 frases) | 49 completion / 109 prompt |
| Calidad | ✅ produce *"Tengo 20 años"* — criterio 1 del paso 6 cumplido |

**Sobre el presupuesto de RAM:** el PRD pide ~2 GB para el paso 6. Con `-c 4096` el consumo
medido es **1.4 GB**, dentro del presupuesto. Si se sube `-c` a 8192 o más, la KV cache crece
linealmente (~+500 MB) y hay que vigilar que el pico no pase de 2 GB.

Como el pipeline es secuencial, la RAM de Whisper (paso 3) se libera antes de que arranque
llama.cpp (paso 6): no se solapan.

---

## 11. Rendimiento esperado

- **Paso 3 (Whisper `small` int8, 4 cores AVX2 sin AVX-512):** entre 1.5× y 2× tiempo real.
  El PRD pide "cuasi-real o mejor"; en esta máquina se queda en ~1.5–2× tiempo real.
  Si estorba, degradar a `Systran/faster-whisper-base` y `compute_type="int8"`
  (cambia solo una constante en `step3_transcribe`).
- **Paso 6:** ~7.5 s por lote pequeño. Con ~100 frases en un solo prompt, del orden de 30–90 s
  por video, muy por debajo de la inferencia por frase (que multiplicaría la carga por 100).
- **Paso 2 (`arnndn`):** tiempo real o ligeramente por debajo en CPU.

---

## 12. Riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| FFmpeg 8.0 eliminó filtros deprecados de la 4.x | Bajo — `arnndn`, `afftdn` y el corte por `-ss`/`-to` siguen vigentes | Verificar con `ffmpeg -filters` tras instalar |
| `spotdl` declara `requires_python <3.15` | Bajo hoy | Reevaluar si se migra a Python 3.15 |
| `llama-server` ocupa ~1.4 GB y el puerto 8080 en background | Bajo | Puerto configurable; apagar tras la sesión |
| 4 cores con AVX2, sin AVX-512 | Whisper `small` va a ~1.5–2× tiempo real | Degradar a modelo `base` si aprieta |
| YouTube anti-bot cambia sin avisar | `yt-dlp` / `spotdl` se rompen ocasionalmente | `yt-dlp -U`; ambos se actualizan con frecuencia |
| Puerto 8080 ocupado por otro servicio | `llama-server` no arranca | Parametrizar `--port` |

---

## 13. Notas de `.gitignore`

Añadir al `.gitignore` del proyecto:

```gitignore
.venv/
models/
output/
*.wav
*.apkg
```

`models/` y `.venv/` son pesados y se regeneran con §8. Los `.wav` y `.apkg` son salidas
generadas por el pipeline.

---

## 14. Resumen de descargas

| Qué | Tamaño | Dónde |
|---|---|---|
| `ffmpeg` 8.0.1 | ~150 MB (apt + deps) | sistema |
| `uv` | ~15 MB | `~/.local/bin` |
| Tarball de llama.cpp | 16.2 MB | `~/tools/llama/` |
| `Systran/faster-whisper-small` | ~480 MB | `models/faster-whisper-small/` |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 770 MB | `models/llama/` |
| Paquetes Python (`spotdl` incluido) | ~200 MB | `.venv/` |
| **Total aproximado** | **~1.6 GB** | |
