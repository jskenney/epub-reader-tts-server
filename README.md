# EPUB reader TTS server

Breeze TTS 2 on your own GPU, for the **Server voice** in
[jskenney/epub-reader](https://github.com/jskenney/epub-reader), with a voice studio for trying text and
creating voices.

**Licence:** the Breeze TTS code is Apache-2.0, but the model weights *and the audio they generate* are
under the BreezeBlue Research and Non-Commercial License. Fine for personal reading; not for a public or
commercial site.

## What runs

`docker-compose.yml` starts two services on the GPU machine:

| Service | Port | What |
|---|---|---|
| `breeze-tts` | 7860 | The Breeze TTS 2 API (`/v1/audio/speech`). Needs the GPU. |
| `studio` | 7861 | The voice studio (a Gradio app) and the voices folder at `/voices/`. No GPU. |

Voices (`voices.json` plus one reference clip per voice) are stored in **`./tts-voices`** next to
`docker-compose.yml`, an ordinary folder bind-mounted into the studio, so they survive rebuilds and are
easy to back up. Set `TTS_VOICES_DIR` to keep them elsewhere.

Your web server proxies three paths on the reader's site to it:

| Site path | Goes to | Used by |
|---|---|---|
| `/tts/` | `breeze-host:7860/` | the reader's speech requests |
| `/tts-voices/` | `breeze-host:7861/voices/` | the reader's voice list and clips |
| `/tts-studio/` | `breeze-host:7861/` | you, in a browser |

The reader shows **Server voice** when `/tts/health` answers and reads its voices from
`/tts-voices/voices.json`. It reads English and Chinese.

## Requirements

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
```

Then add the three proxy paths on your web server: see `nginx.conf.example` or `Caddyfile.example`.
Response buffering must be off for `/tts/` and `/tts-studio/` so audio and progress stream back. Put the
site behind a password so nobody else can use your GPU.

If you open the studio directly at `http://<host>:7861/` instead of through the proxy, set
`STUDIO_ROOT_PATH=` (empty) before `docker compose up -d`.

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
- **Voices:** preview, rename, reorder (the order the reader lists them) and delete voices.

Saved voices show up in the reader's voice list after reloading the reader page.

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

## Files

| File | What |
|---|---|
| `docker-compose.yml` | Builds Breeze TTS from upstream (pinned commit) for Blackwell GPUs and the voice studio |
| `fast-reader.json` | Warmup profile sized for the reader's prompts (mounted over upstream's) |
| `studio/` | The voice studio: `app.py` (Gradio 6.30), `Dockerfile`, `requirements.txt` |
| `nginx.conf.example`, `Caddyfile.example` | Proxy `/tts/`, `/tts-voices/` and `/tts-studio/` on the reader's site |
| `make-voice.sh` | Creates voices from the command line |
| `tts-voices/` | Your voices (created on first run; git-ignored) |
