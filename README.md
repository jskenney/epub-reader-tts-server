# EPUB reader TTS server

Breeze TTS 2 on your own GPU, for the **Server voice** in
[jskenney/epub-reader](https://github.com/jskenney/epub-reader), with a voice studio for trying text and
creating voices.

**Licence:** the Breeze TTS code is Apache-2.0, but the model weights *and the audio they generate* are
under the BreezeBlue Research and Non-Commercial License. Fine for personal reading; not for a public or
commercial site.

## What runs

`docker-compose.yml` starts three services on the GPU machine:

| Service | Port | What |
|---|---|---|
| `breeze-tts` | 7860 | The Breeze TTS 2 API (`/v1/audio/speech`). Needs the GPU. |
| `studio` | 7861 | The voice studio (a Gradio app) and the voices folder at `/voices/`. No GPU. |
| `asr` | (none) | Speech-to-text for the studio: English Whisper large-v3-turbo with word timings. Uses the GPU (about 2 GB) only while transcribing. Only the studio talks to it, so no port is published. |

These are ordinary folders next to `docker-compose.yml`, bind-mounted into the containers, so they survive
rebuilds and are easy to back up:

| Folder | What | Change with |
|---|---|---|
| `./tts-voices` | Voices: `voices.json` plus one reference clip per voice | `TTS_VOICES_DIR` |
| `./tts-sources` | Long recordings uploaded in the studio, with their transcripts, until you delete them | `TTS_SOURCES_DIR` |
| `./asr-models` | The speech-to-text model (about 1.6 GB, downloaded on the first transcription) | `ASR_MODELS_DIR` |

Your web server proxies three paths on the reader's site to it:

| Site path | Goes to | Used by |
|---|---|---|
| `/tts/` | `breeze-host:7860/` | the reader's speech requests |
| `/tts-voices/` | `breeze-host:7861/voices/` | the reader's voice list and clips |
| `/tts-studio/` | `breeze-host:7861/` | you, in a browser |

The reader shows **Server voice** when `/tts/health` answers and reads its voices from
`/tts-voices/voices.json`. It reads English and Chinese.

## Requirements

(For a Mac, see [On a Mac](#on-a-mac-apple-silicon).)

- Linux with an NVIDIA GPU with at least 12 GB of memory (24 GB recommended), a recent driver, and the
  NVIDIA Container Toolkit for Docker.
- `docker-compose.yml` is set up for Blackwell workstation and consumer GPUs (RTX PRO 4000/6000 Blackwell,
  RTX 50xx: `FLASH_ATTN_CUDA_ARCHS=120`). The upstream image is PyTorch 2.9.1 with CUDA 12.8, which
  supports them. For other cards change that value (90 H100, 89 RTX 40xx, 80 A100).

## Set up

On the GPU machine, in a clone of this repository:

```bash
hf download BreezeBlue/Breeze-TTS-2 --local-dir models/Breeze-TTS-2   # about 7.3 GB
docker compose up -d --build        # first build compiles FlashAttention: allow 20-60 minutes
docker compose logs -f              # wait for "fast warmup" and the studio's "Uvicorn running"
curl http://localhost:7860/health
curl http://localhost:7861/health
docker compose exec asr curl -s localhost:7862/health   # speech-to-text (not published on the host)
```

Then add the three proxy paths on your web server: see `nginx.conf.example` or `Caddyfile.example`.
Response buffering must be off for `/tts/` and `/tts-studio/` so audio and progress stream back. Put the
site behind a password so nobody else can use your GPU.

If you open the studio directly at `http://<host>:7861/` instead of through the proxy, set
`STUDIO_ROOT_PATH=` (empty) before `docker compose up -d`.

## On a Mac (Apple Silicon)

Docker on a Mac can't use the GPU, so the `mac/` folder runs the same three services natively instead,
with the same ports and API: the reader, the studio, the proxy examples and `make-voice.sh` work
unchanged. Breeze runs on the GPU through Apple's MLX, using
[Gust](https://github.com/Shoofio/gust-breeze-tts-2-mlx-server), Shoofio's real-time MLX port of the Breeze
server (Apache-2.0), fetched at a pinned commit and patched (`mac/gust.patch`) to answer like the Linux
server. Speech-to-text runs Whisper on the GPU through PyTorch (`mps`).

**Not yet run on a Mac.** The patch passes Gust's own test suite (those that don't need the model), and the
scripts were checked with stand-in tools on Linux, but nothing here has generated audio on a Mac yet.

Needs macOS 14 or later, Apple Silicon with at least 16 GB of memory (Gust uses about 9 GB, Whisper about
2 GB while transcribing), and [Homebrew](https://brew.sh):

```bash
brew install uv ffmpeg
mac/setup.sh                 # fetches Gust, creates the Python environments, downloads the model (~7 GB)
mac/start.sh                 # Breeze on 7860, the studio on 7861, speech-to-text on 127.0.0.1:7862
curl http://localhost:7860/health
curl http://localhost:7861/health
```

`mac/start.sh` runs in the foreground (Ctrl-C stops it) and logs to `mac/logs/`. To start it at login and
restart it if it stops, as Docker's `restart: unless-stopped` does, run `mac/install-service.sh` (and
`mac/install-service.sh --uninstall` to remove it). Settings go in `mac/settings.env`, one `NAME=value` per
line: the folder and speech-to-text settings below, plus

| Setting | Default | What |
|---|---|---|
| `TTS_BIND` | `0.0.0.0` | The address Breeze and the studio listen on. `127.0.0.1` if the web server runs on the same Mac. |
| `TTS_CORS` | `*` | The browser origins Breeze accepts requests from: `*` (any), or your reader's site, e.g. `https://books.example.com`. |
| `ASR_DEVICE` | `mps` | `cpu` if transcription fails on the GPU (much slower). |

What's different from the Linux server:

- **The weights** are Gust's real-time checkpoint (`Shoof/Gust-Breeze-TTS-2-MLX-Realtime`): Breeze TTS 2
  converted to MLX, with the depth decoder in 8-bit. Same model and licence; Gust's README has its speed
  (faster than real time on an M4 Pro) and listening comparisons. The same seed gives slightly different
  audio than on Linux, so audio the reader saved from one is reused for the other if they share an address.
- **No 512-token limit**: there are no CUDA graphs and no `fast-reader.json`. A prompt over the model's
  2,048-token context is refused with a 400 before any audio, instead of failing partway.
- **Errors** are `{"error": "...", "code": "..."}` rather than `{"detail": ...}`; the reader and the studio
  read either. See [API.md](API.md#on-a-mac) for the rest.
- **What the patch changes:** with no `instruction`, Gust used a default one ("Speak clearly and
  naturally."), which turns cloning into directed speech. The patch makes no instruction mean plain speech
  or plain cloning, as upstream Breeze does, and like upstream it then needs `cfg_scale` 1.
- **Updating Gust:** change `GUST_COMMIT` in `mac/setup.sh`, refresh `mac/gust.patch` against it, and run
  `mac/setup.sh` again. Gust's benchmark and model tests run against the patched copy in `mac/gust`
  (`scripts/bench_mac.py`, `.venv/bin/pytest -m mlx` with `BREEZE_MLX_MODEL` set: see its README).

## The voice studio

Open `https://<your site>/tts-studio/`.

- **Try text:** paste any text, choose a saved voice (or describe one), set `cfg_scale` and the seed, and
  generate. With "Read it like the reader" on, it sends one request per paragraph and converts
  parentheses exactly as the reader does, and it reports seconds of audio, generation time and the
  real-time factor, so you can tell whether your GPU keeps up. A saved voice uses its own seed and
  `cfg_scale`, as in the reader.
- **Design a voice:** describe a voice, generate a sample (`cfg_scale` 4 follows the description most
  closely), try other seeds, and **Save as voice**. The sample becomes the voice's reference clip.
- **Record or upload:** record with the microphone or upload a file, trim it, type its exact transcript,
  test it on some text, and save it. Use 5-15 seconds of clean speech. Recording needs the studio to be
  opened over **HTTPS** (or on localhost); uploading works anywhere.
- **From a long recording:** see below.
- **Voices:** preview, rename, reorder (the order the reader lists them) and delete voices.

Saved voices show up in the reader's voice list after reloading the reader page.

### From a long recording

For finding the best few seconds of your own voice in a long recording, such as an audiobook chapter you
narrated or a podcast.

1. **Upload** an audio file: MP3, M4A, M4B, WAV, FLAC or OGG (audio only; files with video are refused,
   though an audiobook's cover picture is fine). It is converted to the mono 24 kHz WAV that voices use and
   kept in `./tts-sources` until you delete it, so you can come back to it. M4B and other files with chapter
   markers get a **Chapter** list for narrowing the transcript down.
2. **Transcribe.** The `asr` service writes down every sentence with its timing, with a progress bar:
   about a minute for 30 minutes of audio on the GPU. The first time it downloads the model (about
   1.6 GB). The transcript is saved with the recording, so this happens once per file.
3. **Click a sentence** to select it, then **Add sentence before / after** or **Drop first / last** to
   grow or shrink the clip. Or type start and end times (or nudge them with the arrows); edges that fall
   inside a word move to just outside it. The transcript box fills in with exactly the words in the clip;
   correct it if the transcription misheard anything, since it must match what is said.
4. **Clean-up** (applied to the clip): even out loudness and fade the edges (both on by default), shorten
   pauses longer than 0.6 s, reduce background noise.
5. **Test this clip** speaks the test text with that clip. Every test is added to **Candidates**, with
   the same text and seed, so differences come from the clip; click one to load it back.
6. **Save the selected clip as a voice** when one sounds right. The voice remembers which file and times
   it came from.

The clip's length is shown as you go: 5-15 s is best, up to 20 s is allowed. A clip and its transcript
share the server's prompt with every paragraph the reader sends (see Fast mode), so longer clips make long
paragraphs fail. Without the `asr` service you can still choose a clip by its times and type the
transcript. **Delete this recording** (under the transcript) removes the upload and its transcript;
voices saved from it are kept.

The proxy has to accept large uploads: `nginx.conf.example` allows 1 GB on `/tts-studio/` (nginx's default
is 1 MB). Caddy has no limit.

These are all the parameters Breeze's API takes: the text, an `instruction` (description), `cfg_scale`,
`seed`, and a reference clip with its transcript. It has no temperature or speed settings; the reader
applies speed at playback.

### Command line

`make-voice.sh` does the design and recording steps without the studio:

```bash
export BREEZE_URL=http://localhost:7860   # run from the folder with docker-compose.yml
./make-voice.sh design narrator "Warm narrator" "A warm, calm middle-aged woman with a clear, steady storytelling voice."
./make-voice.sh clip me "My voice" recording.wav "The exact words spoken in the recording."
```

It writes to `./tts-voices` (or `VOICES_DIR`). Keep the clips out of public repositories: they are model
output under the non-commercial licence.

### Moving voices from the web server

Earlier versions kept voices in `tts-voices/` next to the reader's `index.html`. Copy that folder's
contents (`voices.json` and the `.wav` files) into `./tts-voices` on the GPU machine; the reader now loads
them from `/tts-voices/` on the site root.

## Check the speed

Breeze's own figure (3.1x faster than real time) is on an H100. The studio's **Try text** tab reports the
real-time factor for any text; above 1x keeps up with reading. Or time one paragraph by hand:

```bash
time curl -s -X POST http://localhost:7860/v1/audio/speech \
  -F "text=The old house stood at the end of the lane, its windows catching the last of the evening light. Inside, someone had left a book open on the table, as if they meant to come back to it." \
  -F "instruction=A calm narrator." -o /tmp/test.pcm
python3 -c "import os; print(os.path.getsize('/tmp/test.pcm') / 2 / 24000, 'seconds of audio')"
```

## Fast mode

`docker-compose.yml` runs with `--fast-all`: every stage uses CUDA graphs, captured at start-up for a list
of prompt lengths and then frozen. Upstream's list (`configs/fast.json`) stops at 256 tokens for the
reader's requests, and a long paragraph plus a voice's reference clip goes past it, failing with
`backbone prefill CUDA graph (1, 288) was not declared in the warmup profile`. So the compose file mounts
`fast-reader.json` in its place: upstream's profile with prefill lengths up to 512 tokens and text-encoder
lengths up to 320 at cfg 1 (the reader's requests: a paragraph of up to ~600 characters plus a 5-15 s
reference clip and its transcript come to roughly 350-400 tokens), and without the batch-4 text-encoder
graphs, which only voice direction (reference clip plus instruction) uses.

Every extra graph costs GPU memory: going up to 1024 tokens ran a 24 GB card out of memory at start-up. If
the reader ever sends a prompt longer than the list covers, that request fails and the reader retries the
paragraph one sentence at a time.

Running only the per-frame stages with graphs (`--fast-backbone-decode --fast-depth-decoder --fast-codec`)
avoids the length limit with far less memory, but it was clearly slower on an RTX PRO 4000 Blackwell SFF,
with pauses between paragraphs.

## How the reader uses it

- One request per paragraph (up to about 600 characters), sent one at a time, with the next two
  paragraphs requested ahead. The API runs a single request at a time and answers 409 while busy; the
  reader waits and retries.
- Parentheses and square brackets in the book are turned into dashes before sending, because Breeze
  performs `(sigh)` and `[laugh]` as vocal events.
- Speed is applied at playback with pitch preserved. The highlighted word is estimated, as with the other
  natural voices.

The requests and responses in full, for writing another client: [API.md](API.md).

## Settings

Everything below has a working default. Set the `docker-compose.yml` values in a `.env` file next to it
(one `NAME=value` per line) or in the shell, then run `docker compose up -d` again.

**In `docker-compose.yml`**

| Variable | Default | What it does / when to change it |
|---|---|---|
| `BREEZE_MODEL_DIR` | `./models/Breeze-TTS-2` | Where the Breeze weights are (mounted read-only). Set it if you keep them on another disk. |
| `TTS_VOICES_DIR` | `./tts-voices` | The voices: `voices.json` and the reference clips. |
| `TTS_SOURCES_DIR` | `./tts-sources` | Long recordings uploaded in the studio, with their transcripts. |
| `ASR_MODELS_DIR` | `./asr-models` | The downloaded speech-to-text model. |
| `STUDIO_ROOT_PATH` | `/tts-studio` | The path your web server serves the studio at. Set it to empty (`STUDIO_ROOT_PATH=`) to open the studio directly at `http://<host>:7861/`. |
| `ASR_MODEL` | `openai/whisper-large-v3-turbo` | The Whisper model for speech-to-text. `openai/whisper-large-v3` can be slightly more accurate but is several times slower and needs about twice the GPU memory (roughly 3.5 GB). Keep it English-capable: the service always transcribes as English. |
| `ASR_IDLE_UNLOAD` | `600` | Seconds without a transcription before the model frees its GPU memory. `0` keeps it loaded. |
| `ASR_BATCH` | `8` | Pieces of audio transcribed at once. Lower it (4 or 2) if the GPU runs out of memory while Breeze is also running. |
| `FLASH_ATTN_CUDA_ARCHS` | `120` | Build argument for Breeze: the GPU architecture FlashAttention is compiled for (120 Blackwell workstation/consumer, 90 H100, 89 RTX 40xx, 80 A100). Edit it in `docker-compose.yml` and rebuild. |

The ports (7860 for Breeze, 7861 for the studio) are set in `docker-compose.yml`. The studio reaches Breeze
and `asr` by their service names (`BREEZE_URL`, `ASR_URL`); change those only if you run the services
somewhere else. When running `studio/app.py` or `asr/server.py` outside Docker, their other variables
(`VOICES_DIR`, `SOURCES_DIR`, `STUDIO_PORT`, `ASR_PORT`, `HF_HOME`) are listed at the top of each file.

**In the code**

| Setting | Default | What it does / when to change it |
|---|---|---|
| `fast-reader.json` | prefill up to 512 tokens, text encoder up to 320 at cfg 1 | The prompt lengths Breeze prepares CUDA graphs for at start-up (see [Fast mode](#fast-mode)). A reference clip, its transcript and a paragraph must fit. Larger values use more GPU memory: 1024 ran out of memory on a 24 GB card. |
| `SPAN_MAX` in `studio/app.py` | `600` | Characters per request when **Try text** reads like the reader. It matches the reader's `SPAN_MAX` for the server voice; change both together, and only within `fast-reader.json`'s limits. |
| `CLIP_GOOD`, `CLIP_MAX`, `CLIP_MIN` in `studio/app.py` | `15`, `20`, `3` seconds | The clip lengths **From a long recording** calls good, refuses above and calls too short. Longer clips leave less of the prompt for paragraphs, so raise `CLIP_MAX` only together with `fast-reader.json`. |

## Files

| File | What |
|---|---|
| `API.md` | Every HTTP endpoint: the fields each request takes, the responses, headers and errors |
| `mac/` | Running natively on an Apple Silicon Mac (see [On a Mac](#on-a-mac-apple-silicon)): `setup.sh`, `start.sh`, `install-service.sh`, and `gust.patch`, the changes to Gust |
| `docker-compose.yml` | Builds Breeze TTS from upstream (pinned commit) for Blackwell GPUs and the voice studio |
| `fast-reader.json` | Warmup profile sized for the reader's prompts (mounted over upstream's) |
| `studio/` | The voice studio: `app.py` (Gradio 6.30), `Dockerfile`, `requirements.txt` |
| `asr/` | Speech-to-text for the studio: `server.py` (Whisper through transformers, on the same PyTorch 2.9.1 / CUDA 12.8 base as Breeze), `Dockerfile`, `requirements.txt` |
| `nginx.conf.example`, `Caddyfile.example` | Proxy `/tts/`, `/tts-voices/` and `/tts-studio/` on the reader's site |
| `make-voice.sh` | Creates voices from the command line |
| `tts-voices/` | Your voices (created on first run; git-ignored) |
| `tts-sources/`, `asr-models/` | Uploaded long recordings and the speech-to-text model (created on first use; git-ignored) |
