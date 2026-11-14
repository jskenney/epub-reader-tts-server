# API

Every HTTP endpoint the services in `docker-compose.yml` offer: what each request takes, what comes back, and the
errors. Three services answer HTTP:

| Service | Port | Site path (through the proxy) | Endpoints | Used by |
|---|---|---|---|---|
| `breeze-tts` | 7860 | `/tts/` | [`GET /health`](#get-health), [`GET /v1/model`](#get-v1model), [`POST /v1/audio/speech`](#post-v1audiospeech), [`POST /v1/audio/speech/batch`](#post-v1audiospeechbatch) | the reader, the studio, `make-voice.sh` |
| `studio` | 7861 | `/tts-voices/` → `/voices/`, `/tts-studio/` → `/` | [`GET /voices/voices.json`](#get-voicesvoicesjson), [`GET /voices/<clip>`](#get-voicesclip), [`GET /health`](#get-health-1), the studio UI | the reader (voices), you (UI) |
| `asr` | 7862 (not published) | none | [`GET /health`](#get-health-2), [`POST /transcribe`](#post-transcribe) | the studio only |

Paths below are as the service sees them. Through the proxy, prefix the site path: the reader asks for
`https://<site>/tts/v1/audio/speech`, which the proxy sends to `http://breeze-host:7860/v1/audio/speech`. The services
themselves have no authentication; the proxy's password (see `nginx.conf.example`, `Caddyfile.example`) is what keeps
others off your GPU. All are FastAPI apps, so `breeze-tts` and `asr` also serve generated docs at `/docs` and
`/openapi.json`.

Times are in seconds. Audio sizes assume Breeze's 24 kHz, 16-bit mono output (48,000 bytes per second).

---

## Breeze TTS API (`breeze-tts`, port 7860)

The upstream Breeze TTS 2 server (`python -m breeze_infer.api`, built from
[breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) at commit `daa6cb2`, the one
`docker-compose.yml` pins), run with `--fast-all`. This section describes that commit's `breeze_infer/api.py`.

**One request at a time.** The server holds a lock for the whole of a request, from reading the form until the last
byte of audio is sent (or the client goes away). A second request made meanwhile gets [409](#errors) at once; it is
not queued. Callers queue and retry themselves, as the reader and the studio do.

### `GET /health`

Whether the model is loaded and ready.

| Status | Body | When |
|---|---|---|
| 200 | `{"status": "ok", "sample_rate": 24000}` | Ready. `sample_rate` is the rate of the PCM every speech request returns. |
| 503 | `{"status": "loading"}` | Still starting: loading the weights and, with `--fast-all`, capturing the CUDA graphs (several minutes). |

Content type `application/json`. The reader shows **Server voice** only when this answers 200 with JSON whose
`status` is `"ok"`; anything else (a proxy's HTML error page, a timeout after 5 s) counts as no server.

### `GET /v1/model`

Facts about the loaded checkpoint, for clients that cache audio per model.

```json
{
  "frame_rate": 12.5,
  "model_digest": "3f9c…e1",
  "max_new_tokens": 1500,
  "max_batch_texts": 128
}
```

| Field | Type | Meaning |
|---|---|---|
| `frame_rate` | number | The audio codec's frames per second (`codec_config._frame_rate` in the model's `config.json`). |
| `model_digest` | string | SHA-256 (hex) of the checkpoint's config, index and weight files, worked out once at start-up. Changes only when the weights do. |
| `max_new_tokens` | integer | The most audio frames one text may produce (1500: two minutes at 12.5 frames a second). Longer speech is cut off there. |
| `max_batch_texts` | integer | The most texts [`/v1/audio/speech/batch`](#post-v1audiospeechbatch) takes in one request. |

503 with `{"detail": "API settings are not initialized."}` only if the server was started without its settings
(not the case with `docker-compose.yml`). Not used by the reader or the studio.

### `POST /v1/audio/speech`

Speak one text; the audio streams back while it is generated.

**Request:** `multipart/form-data` (needed when sending `ref_audio`; `application/x-www-form-urlencoded` works
otherwise). Fields:

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `text` | string | yes | | What to say. English or Chinese. See [Text](#text) below. |
| `ref_audio` | file | with `ref_text` | none | A reference clip of the voice to speak in. See [Reference clip](#reference-clip). |
| `ref_text` | string | with `ref_audio` | `""` | The exact words spoken in `ref_audio`. Leading and trailing spaces are dropped. |
| `instruction` | string | no | none | A description of the voice, or (with a reference clip) of how to deliver the text: tone, emotion, pace. Write it in the language of `text`. |
| `cfg_scale` | number | no | `1.0` | Classifier-free guidance: how closely to follow the instruction or the reference. Must be finite and greater than 0. 1 is plain generation; 4 follows an instruction closely (what the studio uses to design voices). |
| `seed` | integer | no | `42` | The random seed. The same text, fields and seed give the same audio on the same build. |

`ref_audio` and `ref_text` go together: both or neither. The fields choose one of four modes:

| `ref_audio` + `ref_text` | `instruction` | Mode | Example use |
|---|---|---|---|
| no | no | Plain speech, in a voice the model picks | |
| no | yes | **Voice design:** a new voice from the description | The studio's **Design a voice** |
| yes | no | **Voice cloning:** speak in the reference clip's voice | The reader, for voices saved with a clip |
| yes | yes | **Voice direction:** the clip's voice, delivered as the instruction says | |

**Response: 200**, the audio, streamed:

| Header | Value |
|---|---|
| `Content-Type` | `audio/pcm` |
| `X-Sample-Rate` | `24000` (the rate, as `GET /health` reports it) |
| `X-Sample-Format` | `s16le` |
| `Cache-Control` | `no-store` |

The body is raw PCM, with no WAV header: mono, signed 16-bit little-endian samples at `X-Sample-Rate`. It has no
`Content-Length` (it is sent in chunks as the model produces audio), and a chunk can end in the middle of a sample, so
keep an odd trailing byte for the next chunk. The length of the body says how long the speech is (bytes ÷ 2 ÷ rate).
To save it as WAV, add a header (`make-voice.sh` and the studio do so with Python's `wave` module) or convert it:

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

Closing the connection early stops generation and frees the server for the next request.

**Errors:** see [Errors](#errors). Also note:

- **A prompt longer than the warmup profile.** With `--fast-all`, the CUDA graphs cover only the prompt lengths in
  `fast-reader.json` (a reference clip, its transcript and the text together: up to 512 tokens to prefill, 320 for the
  text encoder). A longer prompt fails while the audio is being generated, with
  `backbone prefill CUDA graph (1, N) was not declared in the warmup profile` in the server's log. By then the 200
  status and headers have gone out, so the response ends early (empty, or cut short) instead of carrying an error
  status. The reader treats such a short or broken answer as a failure and asks again one sentence at a time. Keep
  paragraphs to about 600 characters with a 5–15 s clip (see the README's *Fast mode*).
- **Too much to say.** Speech stops at `max_new_tokens` frames (two minutes), whatever the text's length.

#### Text

- Breeze performs vocal events written inline: in English in parentheses, `(laugh)`, `(sigh)`, `(cough)`,
  `(clears throat)`; in Chinese in square brackets, `[笑]`, `[叹气]`. Any other parenthesised text is liable to be
  performed too, so the reader turns parentheses and square brackets into dashes before sending: `She paused (sighs)`
  is sent as `She paused — sighs`. (The studio's "Read it like the reader" does the same.)
- There is no speed, pitch or volume field: change speed at playback (the reader does, keeping the pitch).

#### Reference clip

- Any common audio format; the studio and `make-voice.sh` always send mono 24 kHz 16-bit WAV, the format voices are
  stored in, and that is the safe choice.
- 5–15 s of clean speech by one speaker, with `ref_text` matching exactly what is said. Longer clips leave less of the
  prompt for the text (see above): the studio refuses clips over 20 s.
- An empty file is refused (400, `Reference audio is empty.`). A part named `ref_audio` with no file name counts as no
  clip.
- The clip is sent again with every request (the server keeps nothing between requests). Behind nginx, the proxy's
  `client_max_body_size` must allow it (`nginx.conf.example` allows 20 MB).

#### Examples

```bash
# Voice design, saved as WAV
curl -sS -X POST http://localhost:7860/v1/audio/speech \
  -F "text=The old house stood at the end of the lane." \
  -F "instruction=A warm, calm narrator with a clear, steady voice." \
  -F "cfg_scale=4" -F "seed=7" -D headers.txt -o speech.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav

# Cloning a saved voice, as the reader does
curl -sS -X POST http://localhost:7860/v1/audio/speech \
  -F "text=It was a bright cold day in April." \
  -F "ref_audio=@tts-voices/narrator.wav" \
  -F "ref_text=The exact words spoken in narrator.wav." \
  -F "cfg_scale=1" -F "seed=42" -o speech.pcm
```

```js
// In a browser: play the stream once it has arrived
const form = new FormData();
form.append('text', 'It was a bright cold day in April.');
form.append('ref_audio', clipBlob, 'narrator.wav');
form.append('ref_text', 'The exact words spoken in narrator.wav.');
form.append('seed', '42');
const r = await fetch('/tts/v1/audio/speech', { method: 'POST', body: form });
if (r.status === 409) { /* busy: wait and try again */ }
const rate = Number(r.headers.get('X-Sample-Rate')) || 24000;
const pcm = new Int16Array(await r.arrayBuffer());   // (read r.body to start playing sooner)
const ctx = new AudioContext(), buf = ctx.createBuffer(1, pcm.length, rate);
buf.getChannelData(0).set(Float32Array.from(pcm, s => s / 32768));
const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination); src.start();
```

### `POST /v1/audio/speech/batch`

Speak several texts in one go, in the same voice, and get them back together when all are done. Upstream offers it
for offline work; nothing in this repository uses it, and it hasn't been tried with this setup's `--fast-all` warmup.
It runs the model without CUDA graphs (batched), so it doesn't share the prompt-length limit above, but it is slower
for a single text.

**Request:** `multipart/form-data`, the same fields as [`/v1/audio/speech`](#post-v1audiospeech) except that `texts`
replaces `text`, and one more:

| Field | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `texts` | string | yes | | A JSON array of 1–128 non-empty strings, for example `["First paragraph.", "Second paragraph."]`. |
| `max_new_tokens` | integer | no | `1500` | The most frames for each text; clamped to 1–1500. |

`instruction`, `cfg_scale`, `ref_audio` with `ref_text`, and `seed` apply to every text.

**Response: 200**, not streamed:

| Header | Value |
|---|---|
| `Content-Type` | `application/octet-stream` |
| `X-Segment-Bytes` | The byte length of each text's audio, in order, comma-separated: `96000,143520` |
| `X-Sample-Rate` | `24000` |
| `X-Sample-Format` | `s16le` |
| `Cache-Control` | `no-store` |

The body is every text's PCM (as above), one after another in the order of `texts`; split it with `X-Segment-Bytes`.

**Errors:** as for `/v1/audio/speech`, plus 400 for `texts`: `texts must be a JSON array of strings: …`,
`texts must be a non-empty JSON array.`, `texts has N entries; the limit is 128.`,
`Every entry in texts must be a non-empty string.`

### Errors

Errors from Breeze are JSON, FastAPI's form:

| Status | Body | Cause |
|---|---|---|
| 400 | `{"detail": "cfg_scale must be greater than 0."}` | `cfg_scale` is 0, negative, infinite or NaN. |
| 400 | `{"detail": "ref_audio and ref_text must be provided together or both omitted."}` | One without the other (a `ref_text` of only spaces counts as none). |
| 400 | `{"detail": "Reference audio is empty."}` | `ref_audio` has no bytes. |
| 409 | `{"detail": "An inference request is already running."}` | Another request holds the server (including one whose client has just gone away and is still being stopped). Wait and try again. |
| 422 | `{"detail": [{"type": "missing", "loc": ["body", "text"], "msg": "Field required", …}]}` | A required field is missing, or a field isn't the right type (`cfg_scale` not a number, `seed` not an integer). `detail` is a list, one entry per problem. |
| 500 | `Internal Server Error` (plain text) | The reference clip couldn't be read, or the model failed while preparing the request. |
| 503 | `{"status": "loading"}` (`/health` only) | Still starting. |

Through a proxy, a stopped `breeze-tts` container shows up as 502, 503 or 504 from the proxy itself, often with an
HTML page: the reader treats those (and any HTML answer) as "the server is gone" and hides Server voice until
`/health` answers again.

---

## Voices (`studio`, port 7861)

The studio serves the voices folder (`./tts-voices` on the host, `VOICES_DIR` in the container) read-only at
`/voices/`; the site proxies `/tts-voices/` to it. The studio, `make-voice.sh` and you (by hand) write it. Responses
from `/voices/` carry `Cache-Control: no-cache`, so a new or changed voice shows up as soon as the reader asks again.

### `GET /voices/voices.json`

The voices the reader offers, in the order it lists them.

```json
{
  "voices": [
    {
      "id": "narrator",
      "name": "Warm narrator",
      "ref_audio": "narrator.wav",
      "ref_text": "The old house stood at the end of the lane, its windows catching the last of the evening light.",
      "seed": 42,
      "cfg_scale": 1.0,
      "description": "A warm, calm middle-aged woman with a clear, steady storytelling voice."
    },
    {
      "id": "me",
      "name": "My voice",
      "ref_audio": "me.wav",
      "ref_text": "The exact words spoken in the clip.",
      "seed": 42,
      "cfg_scale": 1.0,
      "source": "chapter-03.m4b 12:04.5–12:16.0"
    },
    {
      "id": "described",
      "name": "Described only",
      "instruction": "A deep, slow, gravelly male voice.",
      "seed": 7,
      "cfg_scale": 1.0
    }
  ]
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `voices` | array | yes | The voices, in the order the reader lists them (the studio's **Voices** tab reorders them). |
| `id` | string | yes | Unique; `a-z`, `0-9`, `-`, `_` (the studio and `make-voice.sh` insist). The reader remembers the chosen voice by it. |
| `name` | string | no | Shown in the reader's voice list; `id` if missing. |
| `ref_audio` | string | with `ref_text` | The reference clip's file name, relative to the voices folder. Sent as [`ref_audio`](#post-v1audiospeech) with every request. |
| `ref_text` | string | with `ref_audio` | The clip's exact transcript, sent as `ref_text`. |
| `instruction` | string | without a clip | A description of the voice, sent as `instruction`. A voice needs a clip with its transcript, or an instruction. (One with both is sent both: voice direction.) |
| `seed` | integer | no | Sent as `seed`; 42 if missing. |
| `cfg_scale` | number | no | Sent as `cfg_scale`; 1 if missing. |
| `description` | string | no | The description a designed voice was made from (kept for you; not sent). |
| `source` | string | no | The recording and times a clip was cut from (kept for you; not sent). |

Other fields are kept and ignored. How the reader reads the list:

- A voice is offered if it has `ref_audio` with `ref_text`, or an `instruction`; others are left out.
- `ref_audio` must stay inside the voices folder: a name that resolves elsewhere (`../x.wav`, another site, an
  encoded slash) isn't fetched. The voice then falls back to its `instruction` if it has one, and is left out
  otherwise.
- The reader caches the speech it makes per voice, keyed by the server address, `ref_audio` and its version (below),
  `ref_text`, `instruction`, `cfg_scale` and `seed`. Changing any of those makes new speech; changing `name`, the
  order, or other fields doesn't.

| Status | When |
|---|---|
| 200 | `application/json`, as above. |
| 404 | No voices yet (no `voices.json`). The reader says so and points to the studio. |

### `GET /voices/<clip>`

A reference clip named in `voices.json`, for example `GET /voices/narrator.wav`. The clips the studio and
`make-voice.sh` write are WAV, mono, 24 kHz, 16-bit. Served as static files: `HEAD` works, and responses carry `ETag`
and `Last-Modified`. The reader fetches each clip once per session and sends it with every request in that voice;
it asks with `HEAD` for the clip's version (its `ETag`, or date and size), so replacing a clip under the same name
makes its cached speech be made again. 404 if the file isn't there.

### `GET /health`

`{"status": "ok", "voices": 3}`: the studio is up, and `voices` is how many entries `voices.json` has (0 without
one). Used by the container's health check. 500 if `voices.json` exists but isn't valid JSON.

### The studio UI (`/`)

The Gradio app (served at `/tts-studio/` through the proxy). Gradio's own endpoints under it (`/gradio_api/…`, the
upload and queue routes) belong to the UI and change with Gradio's version: they aren't an API to build on. Use
[`/v1/audio/speech`](#post-v1audiospeech) and the voices folder instead.

---

## Speech-to-text (`asr`, port 7862)

English Whisper (`ASR_MODEL`, default `openai/whisper-large-v3-turbo`) with word timings. Only the studio calls it,
over the compose network at `http://asr:7862`; the port isn't published on the host. From the host:
`docker compose exec asr curl -s localhost:7862/health`.

The model is loaded on the first transcription (downloaded the first time, about 1.6 GB) and unloaded after
`ASR_IDLE_UNLOAD` seconds without one (600; 0 keeps it). One transcription runs at a time; others wait.

### `GET /health`

```json
{"status": "ok", "model": "openai/whisper-large-v3-turbo", "device": "cuda", "loaded": false}
```

| Field | Meaning |
|---|---|
| `status` | Always `"ok"` while the service runs. |
| `model` | The Whisper model it uses. |
| `device` | `"cuda"`, or `"cpu"` without a GPU (much slower). |
| `loaded` | Whether the model is in memory now. |

### `POST /transcribe`

Transcribe a recording; progress streams back as it goes.

**Request:** `multipart/form-data` with one field:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `file` | file | yes | Any audio ffmpeg can read (MP3, M4A, M4B, WAV, FLAC, OGG, …), any length (up to 30 minutes of ffmpeg time to decode it). The file name's extension is used as a hint. Video is ignored. |

**Response: 200**, `application/x-ndjson`: one JSON object per line, streamed. Lines come in this order:

1. Zero or more **progress** lines while it works:

   ```json
   {"progress": 0.0, "status": "Reading the audio…"}
   {"progress": 0.0, "status": "Waiting for another transcription to finish…"}
   {"progress": 0.0, "status": "Loading openai/whisper-large-v3-turbo (the first time downloads it)…"}
   {"progress": 0.4213, "status": "Transcribed 12.6 of 30.0 minutes, about 2 min left"}
   ```

   `progress` goes from 0 to 1 (how far through the audio, by time); `status` is a sentence to show.

2. Then exactly one of:

   - a **result** line, the last line:

     ```json
     {"result": {
       "duration": 1800.04,
       "model": "openai/whisper-large-v3-turbo",
       "words": [
         {"start": 0.52, "end": 0.81, "text": "It"},
         {"start": 0.81, "end": 0.98, "text": "was"},
         {"start": 0.98, "end": 1.44, "text": "bright."}
       ],
       "segments": [
         {"start": 0.52, "end": 1.44, "first": 0, "last": 2, "text": "It was bright."}
       ]
     }}
     ```

   - or an **error** line, the last line: `{"error": "The recording is empty."}`, or
     `{"error": "<ExceptionType>: <message>"}` (ffmpeg couldn't read the file, the GPU ran out of memory, …).

Errors come as an `error` line in a 200 response, since the status has been sent before the work starts; check every
line for `error`. Only a request without `file` gets a non-200 answer (422, FastAPI's validation error).

| Result field | Type | Meaning |
|---|---|---|
| `duration` | number | The recording's length, in seconds. |
| `model` | string | The model that transcribed it. |
| `words` | array | Every word, in order. |
| `words[].start`, `words[].end` | number | When the word begins and ends, in seconds from the start (3 decimals). `end` ≥ `start`; a missing end is filled in (the next word's start, at most 0.6 s on). |
| `words[].text` | string | The word with its punctuation attached (`"bright."`), no surrounding spaces. |
| `segments` | array | The words grouped into sentences, in order. |
| `segments[].start`, `segments[].end` | number | The sentence's first word's start and last word's end. |
| `segments[].first`, `segments[].last` | integer | Indexes into `words` of the sentence's first and last word (inclusive). |
| `segments[].text` | string | The sentence's words joined by spaces. |

How it gets there, which explains the shape:

- The audio is decoded to 16 kHz mono and cut into pieces of 18–28 s at the quietest 50 ms near each cut, so a piece
  never splits a word; pieces that are nearly silent are skipped (Whisper invents words for silence), so stretches of
  silence or music have no words.
- Pieces are transcribed `ASR_BATCH` at a time, always as English. A piece Whisper wrote without punctuation is
  transcribed again with a punctuated prompt, and its punctuation and capitals are copied onto the words where the two
  agree.
- If the model can't give word timings, it falls back to sentence timings and spreads each sentence's words evenly
  over it.
- Sentences end at `.`, `!`, `?` or `…` (not after `Mr.`, `Dr.`, `St.` and the like), or failing that at a pause:
  over 1 s, over 0.5 s once a sentence is 8 s long, or the longest gap among its last words once it is 15 s long.

```bash
docker compose exec asr sh -c 'curl -sS -N -F "file=@/tmp/chapter.m4b" localhost:7862/transcribe'
```

---

## How the reader uses the API

The reader ([jskenney/epub-reader](https://github.com/jskenney/epub-reader)) takes the two addresses from its
`TTS_SERVER` setting (default `{"api": "/tts/", "voices": "/tts-voices/"}` on the site root). In order:

1. **Is there a server?** `GET /tts/health`, with a 5 s time limit, when the read-aloud bar opens and before
   playing, at most every 15 s. Only a 200 JSON answer with `"status": "ok"` counts. The reader offers Server voice
   for books in English or Chinese.
2. **Which voices?** `GET /tts-voices/voices.json` (a 502, 503 or 504, or a 200 that isn't JSON, counts as no
   server), then each voice's clip from `/tts-voices/<ref_audio>` (once per
   session, plus a `HEAD` for its version).
3. **Speech.** One `POST /tts/v1/audio/speech` per paragraph, as multipart: `text` (parentheses and brackets turned
   into dashes), the voice's `ref_audio` (the clip, named as in `voices.json`) and `ref_text`, or its `instruction`,
   then `cfg_scale` and `seed` (defaults 1 and 42). A paragraph longer than 600 characters (`SPAN_MAX.server`) is
   split at sentence ends. Requests go one at a time, with the next two paragraphs asked for ahead.
4. **Busy.** A 409 is asked again after 150 ms, doubling to at most 1.5 s, up to 40 times (about a minute).
5. **Reading the answer.** The stream is read as it arrives, at `X-Sample-Rate` (24000 if missing). The reader gives
   up on a request that hasn't started answering after 60 s, or whose stream stops for 180 s before any audio or 60 s
   after some. Since the stream has no length, an answer with less than 0.12 s of audio per word of `text` counts as
   cut off (the server failed partway) and fails.
6. **Failures.** 502, 503, 504, an HTML answer, or no connection: the server counts as gone, and Server voice is
   hidden until `/health` answers again. Another error status: that paragraph fails (a 409 after all the retries, or a
   5xx, is retried later as a passing failure), and a paragraph that keeps failing is asked for one sentence at a
   time.

Speed is applied at playback, with pitch kept; the word highlighted while it speaks is estimated from the text, not
from the server.

## Proxy requirements

What the proxy must let through for the API to work (both examples do):

| Path | Needs |
|---|---|
| `/tts/` | No response buffering (audio streams back as it is generated); request bodies of at least 20 MB (the reference clip comes with every request); a read timeout of a few minutes. |
| `/tts-voices/` | Nothing special: small static files. |
| `/tts-studio/` | WebSocket upgrade, no buffering (progress streams back), request bodies up to the largest recording you upload (1 GB in `nginx.conf.example`; Caddy has no limit), and HTTPS for microphone recording. |
