"""Speech-to-text for the voice studio: English Whisper with word timestamps, on the GPU.

The studio sends a recording (any format ffmpeg reads) and gets back every word with its start and end
time, plus the words grouped into sentences. The recording is cut into pieces of 18-28 seconds at the
quietest moment (a pause between words), so each piece fits Whisper's 30-second window whole: no
stitching of overlapping windows, which repeats or drops words at the seams. Pieces are transcribed
ASR_BATCH at a time and progress is streamed back after each batch.

  POST /transcribe   multipart field "file"; answers with NDJSON lines:
                       {"progress": 0.42, "status": "..."}   (several)
                       {"result": {"duration": s, "words": [{"start", "end", "text"}], "segments": [...]}}
                     or {"error": "..."}
  GET  /health       {"status": "ok", "model", "device", "loaded"}

The model is loaded on the first request and unloaded after ASR_IDLE_UNLOAD seconds without one, so it
only holds GPU memory (about 2 GB for large-v3-turbo) while it is being used.

Environment:
  ASR_MODEL          Hugging Face model id (default openai/whisper-large-v3-turbo)
  ASR_IDLE_UNLOAD    seconds before unloading an unused model (default 600; 0 keeps it loaded)
  ASR_BATCH          pieces transcribed together (default 8 on the GPU, 1 on the CPU)
  ASR_DEVICE         cuda, mps (an Apple Silicon Mac's GPU) or cpu (default: the first available)
  ASR_HOST           the address to listen on (default 0.0.0.0; mac/start.sh uses 127.0.0.1)
  ASR_PORT           default 7862
  HF_HOME            where downloaded models are kept (mounted as a volume in docker-compose.yml)
"""

import difflib
import gc
import json
import os
import re
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
IDLE = float(os.environ.get("ASR_IDLE_UNLOAD", "600"))
PORT = int(os.environ.get("ASR_PORT", "7862"))
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
DEVICE = os.environ.get("ASR_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH = int(os.environ.get("ASR_BATCH", "8" if DEVICE != "cpu" else "1"))
SR = 16000
ENGLISH_ONLY = MODEL.endswith(".en")  # these models take no language setting

state = {"pipe": None, "used": 0.0, "word_times": True}
lock = threading.Lock()  # one transcription at a time
app = FastAPI()


def load():
    if state["pipe"] is None:
        from transformers import pipeline

        state["pipe"] = pipeline("automatic-speech-recognition", model=MODEL, device=DEVICE,
                                 torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32)
    state["used"] = time.time()
    return state["pipe"]


def unload_when_idle():
    while True:
        time.sleep(30)
        if IDLE <= 0:
            continue
        with lock:
            if state["pipe"] is not None and time.time() - state["used"] > IDLE:
                state["pipe"] = None
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif DEVICE == "mps":
                    torch.mps.empty_cache()


def decode(path: str) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, timeout=1800)
    if r.returncode:
        raise ValueError(r.stderr.decode(errors="replace")[:300] or "ffmpeg couldn't read the audio")
    return np.frombuffer(r.stdout, np.float32)


PIECE_MIN, PIECE_MAX = 18.0, 28.0  # seconds; Whisper hears 30 s at a time


def pieces(audio: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) sample ranges of 18-28 s, each ending at the quietest 50 ms in that range."""
    n, win = len(audio), int(0.05 * SR)
    cuts = [0]
    while n - cuts[-1] > PIECE_MAX * SR:
        lo, hi = cuts[-1] + int(PIECE_MIN * SR), cuts[-1] + int(PIECE_MAX * SR)
        seg = audio[lo:hi]
        frames = len(seg) // win
        energy = (seg[:frames * win].reshape(frames, win) ** 2).mean(axis=1)
        cuts.append(lo + int(energy.argmin()) * win + win // 2)
    cuts.append(n)
    out = []
    for a, b in zip(cuts, cuts[1:]):
        # Skip silent pieces: Whisper tends to invent words ("Thank you.") for silence.
        if b - a > SR * 0.2 and np.sqrt(np.mean(audio[a:b] ** 2)) > 0.002:
            out.append((a, b))
    return out


# Whisper sometimes writes a piece without punctuation or capitals ("he hoped there would be stew for
# dinner turnips and…"). Such pieces are transcribed again with a punctuated prompt, which sets the style,
# and the punctuated words replace the plain ones where the two transcripts agree. (A prompt can't be used
# in the word-timing pass itself: transformers then needs gigabytes for the timing alignment.)
STYLE_PROMPT = "Hello. This is a recording of someone reading aloud, with proper punctuation."
_norm = lambda w: re.sub(r"[^\w']", "", w.lower())


def unpunctuated(words: list[dict]) -> bool:
    return len(words) >= 8 and not any(re.search(r"[.,!?;:]", w["text"]) for w in words)


def restyle(pipe, audio: np.ndarray, words: list[dict], gen: dict) -> None:
    try:
        ids = pipe.tokenizer.get_prompt_ids(STYLE_PROMPT, return_tensors="pt")
        text = pipe(audio.copy(), generate_kwargs={**gen, "prompt_ids": ids})["text"]
    except Exception as exc:
        print(f"Couldn't restyle a piece: {exc}", flush=True)
        return
    styled = text.split()
    match = difflib.SequenceMatcher(None, [_norm(w["text"]) for w in words], [_norm(t) for t in styled], autojunk=False)
    for tag, i1, i2, j1, j2 in match.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                words[i1 + k]["text"] = styled[j1 + k]


def run_batch(pipe, audio: np.ndarray, batch: list[tuple[int, int]]) -> list[dict]:
    gen = {} if ENGLISH_ONLY else {"language": "en", "task": "transcribe"}
    inputs = [audio[a:b].copy() for a, b in batch]
    if state["word_times"]:
        try:
            outs = pipe(inputs, return_timestamps="word", batch_size=BATCH, generate_kwargs=gen)
            words = []
            for (a, _), seg, out in zip(batch, inputs, outs):
                piece = [{"start": a / SR + (c["timestamp"][0] or 0.0),
                          "end": None if c["timestamp"][1] is None else a / SR + c["timestamp"][1],
                          "text": c["text"].strip()} for c in out.get("chunks", []) if c["text"].strip()]
                if unpunctuated(piece):
                    restyle(pipe, seg, piece, gen)
                words += piece
            return words
        except Exception as exc:  # a model without alignment heads: fall back to sentence times
            print(f"Word timestamps unavailable ({exc}); using sentence timestamps", flush=True)
            state["word_times"] = False
    outs = pipe(inputs, return_timestamps=True, batch_size=BATCH, generate_kwargs=gen)
    words = []
    for (a, b), out in zip(batch, outs):
        for c in out.get("chunks", []):
            s, e = c["timestamp"]
            s, e = a / SR + (s or 0.0), a / SR + (e if e is not None else (b - a) / SR)
            ws = c["text"].split()
            for i, w in enumerate(ws):  # spread the words evenly over the sentence
                words.append({"start": s + (e - s) * i / len(ws), "end": s + (e - s) * (i + 1) / len(ws), "text": w})
    return words


def tidy(words: list[dict], duration: float) -> list[dict]:
    """Fill in missing end times and keep the times in order."""
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else duration
        if w["end"] is None or w["end"] < w["start"]:
            w["end"] = min(nxt, w["start"] + 0.6)
        w["start"], w["end"] = round(w["start"], 3), round(min(w["end"], duration), 3)
    return words


ABBREV = re.compile(r"\b(mr|mrs|ms|dr|st|vs|mt|jr|sr|prof|etc)\.$", re.I)


def sentences(words: list[dict]) -> list[dict]:
    """Group words into sentences: at sentence-ending punctuation, or failing that at pauses: over a second,
    over half a second once 8 s long, and the longest pause in the last few seconds once 15 s long."""
    segs, cur = [], []

    def flush():
        if cur:
            segs.append({"start": cur[0]["start"], "end": cur[-1]["end"], "first": cur[0]["i"], "last": cur[-1]["i"],
                         "text": " ".join(w["text"] for w in cur)})
            cur.clear()

    for i, w in enumerate(words):
        if cur:
            gap, length = w["start"] - cur[-1]["end"], w["end"] - cur[0]["start"]
            if gap > 1.0 or (gap > 0.5 and length > 8):
                flush()
            elif length > 15:  # no punctuation or pause: split at the longest gap among the last words
                tail = range(max(1, len(cur) - 12), len(cur))
                k = max(tail, key=lambda j: cur[j]["start"] - cur[j - 1]["end"]) if len(cur) > 1 else len(cur)
                rest = cur[k:]
                del cur[k:]
                flush()
                cur.extend(rest)
        cur.append({**w, "i": i})
        if re.search(r"[.!?…][\"'”’)\]]*$", w["text"]) and not ABBREV.search(w["text"]):
            flush()
    flush()
    return segs


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "model": MODEL, "device": DEVICE, "loaded": state["pipe"] is not None})


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1] or ".audio")
    with os.fdopen(fd, "wb") as out:
        while chunk := file.file.read(1 << 20):
            out.write(chunk)

    def work():
        line = lambda obj: json.dumps(obj) + "\n"
        try:
            yield line({"progress": 0.0, "status": "Reading the audio…"})
            audio = decode(path)
            duration = len(audio) / SR
            if duration < 0.3:
                yield line({"error": "The recording is empty."})
                return
            if lock.locked():
                yield line({"progress": 0.0, "status": "Waiting for another transcription to finish…"})
            with lock:
                if state["pipe"] is None:
                    yield line({"progress": 0.0, "status": f"Loading {MODEL} (the first time downloads it)…"})
                pipe = load()
                todo, words, t0 = pieces(audio), [], time.time()
                for k in range(0, len(todo), BATCH):
                    batch = todo[k:k + BATCH]
                    words += run_batch(pipe, audio, batch)
                    state["used"] = time.time()
                    b = batch[-1][1]
                    done = b / len(audio)
                    left = (time.time() - t0) / done * (1 - done)
                    yield line({"progress": round(done, 4),
                                "status": f"Transcribed {b / SR / 60:.1f} of {duration / 60:.1f} minutes"
                                          + (f", about {left / 60:.0f} min left" if left > 90 else "")})
            words = tidy(words, duration)
            yield line({"result": {"duration": duration, "model": MODEL, "words": words, "segments": sentences(words)}})
        except Exception as exc:
            yield line({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            os.unlink(path)

    return StreamingResponse(work(), media_type="application/x-ndjson")


if __name__ == "__main__":
    threading.Thread(target=unload_when_idle, daemon=True).start()
    print(f"Speech-to-text: {MODEL} on {DEVICE}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
