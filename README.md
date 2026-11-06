# EPUB reader TTS server

Breeze TTS 2 on your own GPU, for the **Server voice** in
[jskenney/epub-reader](https://github.com/jskenney/epub-reader).

The reader's Server voice reads books with [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
running on a GPU machine behind the website's reverse proxy. The reader talks to it at `/tts/` on the site root and
finds voices in `tts-voices/voices.json` next to `index.html`. The option appears in the read-aloud bar
only when `/tts/health` answers. It reads English and Chinese.

**Licence:** the Breeze TTS code is Apache-2.0, but the model weights *and the audio they generate* are
under the BreezeBlue Research and Non-Commercial License. Fine for personal reading; not for a public or
commercial site.

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
docker compose logs -f              # wait for the API to report it is running
curl http://localhost:7860/health   # on the Breeze server
```

The API listens on port 7860 on all interfaces. On your web server, proxy `/tts/` on the site root to
`http://<breeze-host>:7860/`: see `nginx.conf.example` or `Caddyfile.example`. Response buffering must
be off so audio streams back as it is generated.

## Voices

Each voice is a short reference clip plus its exact transcript. The reader sends them with every request,
which keeps the voice the same from paragraph to paragraph.

`make-voice.sh` writes into the reader's `tts-voices/` folder, so run it where that folder is (usually the
web server; it only needs `curl` and `python3`), pointing it at the Breeze API:

```bash
export BREEZE_URL=http://breeze-host:7860
export VOICES_DIR=/srv/www/reader/tts-voices   # the tts-voices/ folder next to the reader's index.html

# Design a voice from a description (generated once, then reused):
./make-voice.sh design narrator "Warm narrator" "A warm, calm middle-aged woman with a clear, steady storytelling voice."

# Or use your own clean 5-15 second recording and its exact transcript:
./make-voice.sh clip me "My voice" recording.wav "The exact words spoken in the recording."
```

Clips go in `VOICES_DIR`, listed in `voices.json` in the order they appear in the reader. Listen to a designed clip before using it; if it isn't right, run again with another seed
(`VOICE_SEED=7 ./make-voice.sh design ...`). Keep the clips out of public repositories: they are model
output under the non-commercial licence.

## Check the speed first

Breeze's own figure (3.1x faster than real time) is on an H100. A small-form-factor card is much slower,
and if generation falls behind playback the reader pauses between paragraphs and says so. After the
server is up, time one paragraph:

```bash
time curl -s -X POST http://breeze-host:7860/v1/audio/speech \
  -F "text=The old house stood at the end of the lane, its windows catching the last of the evening light. Inside, someone had left a book open on the table, as if they meant to come back to it." \
  -F "instruction=A calm narrator." -o /tmp/test.pcm
python3 -c "import os; print(os.path.getsize('/tmp/test.pcm') / 2 / 24000, 'seconds of audio')"
```

If the time is shorter than the seconds of audio, it keeps up.

## Fast mode

`docker-compose.yml` turns on CUDA graphs for the stages that run on every audio frame
(`--fast-backbone-decode --fast-depth-decoder --fast-codec`), which is where the speed comes from. It does
**not** use `--fast-all`: that also captures text-encoder and prefill graphs for a fixed list of prompt
lengths (up to 256 tokens at cfg 1, from upstream's `configs/fast.json`) and freezes them, and a long
paragraph plus the voice's reference clip goes past that, failing with
`backbone prefill CUDA graph (1, 288) was not declared in the warmup profile`. Those two stages run once
per request, so running them eagerly costs little.

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
| `docker-compose.yml` | Builds Breeze TTS from upstream (pinned commit) for Blackwell GPUs and serves it on port 7860 |
| `nginx.conf.example`, `Caddyfile.example` | Reverse-proxy `/tts/` on the reader's site to the Breeze server |
| `make-voice.sh` | Creates voices (designed or from a recording) in the reader's `tts-voices/` folder |
