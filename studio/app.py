"""Voice studio for the EPUB reader's Server voice.

A Gradio app next to the Breeze TTS 2 API that lets you:
  - try any text with a saved voice or a description, with the API's parameters,
  - design voices from a description and save them,
  - record or upload a reference clip, test it, and save it as a voice,
  - pick a clip out of a long recording (an audiobook chapter, a podcast) using its transcript, try
    several, and save the best,
  - preview, rename, reorder and delete voices.

It also serves the voices folder (voices.json plus the reference clips) at /voices/, which the reader
fetches through the website's /tts-voices/ proxy. Voices are stored in VOICES_DIR, a bind-mounted folder
on the host (see docker-compose.yml).

Environment:
  BREEZE_URL         the Breeze API (default http://breeze-tts:7860)
  VOICES_DIR         where voices are stored (default /data/voices)
  SOURCES_DIR        where uploaded long recordings are kept until deleted (default /data/sources)
  ASR_URL            the speech-to-text service (default http://asr:7862)
  STUDIO_ROOT_PATH   the path the studio is served at behind a proxy, for example /tts-studio
                     ("" when opened directly on its port)
  STUDIO_PORT        default 7861
"""

import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BREEZE = os.environ.get("BREEZE_URL", "http://breeze-tts:7860").rstrip("/")
VOICES = Path(os.environ.get("VOICES_DIR", "/data/voices"))
ROOT_PATH = os.environ.get("STUDIO_ROOT_PATH", "").rstrip("/")
PORT = int(os.environ.get("STUDIO_PORT", "7861"))
SOURCES = Path(os.environ.get("SOURCES_DIR", "/data/sources"))
ASR = os.environ.get("ASR_URL", "http://asr:7862").rstrip("/")
MANIFEST = VOICES / "voices.json"
WORK = Path(tempfile.mkdtemp(prefix="studio-"))
VOICES.mkdir(parents=True, exist_ok=True)
SOURCES.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXT = (
    "The old house stood at the end of the lane, its windows catching the last of the evening light. "
    "Inside, someone had left a book open on the table, as if they meant to come back to it."
)
TRY_TEXT = (
    "It was a bright cold day in April, and the clocks were striking thirteen. She paused (sighs) and "
    "looked away.\n\nThe rain had not stopped all day. Somewhere below, a door closed, and the house was quiet again."
)
NO_VOICE = "(no saved voice: use the description)"
ID_RE = re.compile(r"^[a-z0-9_-]+$")
SPAN_MAX = 600  # the reader sends paragraphs of up to about this many characters
_lock = threading.Lock()  # the Breeze API runs one request at a time


# ---------- voices.json ----------

def load_voices() -> list[dict]:
    try:
        return json.loads(MANIFEST.read_text("utf-8")).get("voices", [])
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise gr.Error(f"voices.json couldn't be read: {exc}")


def save_voices(voices: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"voices": voices}, indent=2, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(MANIFEST)


def voice_choices() -> list[tuple[str, str]]:
    return [(f"{v.get('name') or v['id']} ({v['id']})", v["id"]) for v in load_voices()]


def find_voice(vid: str) -> dict | None:
    return next((v for v in load_voices() if v["id"] == vid), None)


def add_voice(entry: dict) -> str:
    voices = load_voices()
    replaced = any(v["id"] == entry["id"] for v in voices)
    voices = [v for v in voices if v["id"] != entry["id"]] + [entry]
    save_voices(voices)
    return f"{'Replaced' if replaced else 'Saved'} **{entry['name']}** ({entry['id']}). It's in the reader's voice list now (reload the reader page)."


def check_id(vid: str, name: str) -> tuple[str, str]:
    vid = (vid or "").strip().lower()
    if not ID_RE.match(vid):
        raise gr.Error("The id may only use a-z, 0-9, - and _ (for example: narrator).")
    return vid, (name or "").strip() or vid


# ---------- audio ----------

def write_wav(pcm: bytes, rate: int, path: Path) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


def wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def to_wav(src: str, dst: Path) -> Path:
    """Convert any recording or upload to mono 24 kHz WAV."""
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "24000", str(dst)],
                       check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise gr.Error(f"That audio couldn't be converted: {exc.stderr.decode(errors='replace')[:300]}")
    return dst


def scratch(suffix: str = ".wav") -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix, dir=WORK)
    os.close(fd)
    return Path(name)


# ---------- the Breeze API ----------

def breeze(text: str, cfg: float, seed: int, instruction: str | None = None,
           ref_audio: Path | None = None, ref_text: str | None = None) -> tuple[bytes, int, float]:
    """One request, as the reader makes them. Returns (s16le PCM, sample rate, generation seconds)."""
    data = {"text": text, "cfg_scale": str(cfg), "seed": str(int(seed))}
    if instruction:
        data["instruction"] = instruction
    if ref_audio:
        data["ref_text"] = ref_text or ""
    # Always multipart, as the reader and make-voice.sh send it.
    files = {k: (None, v) for k, v in data.items()}
    if ref_audio:
        files["ref_audio"] = (ref_audio.name, ref_audio.read_bytes(), "audio/wav")
    with _lock:
        for _ in range(40):
            t0 = time.time()
            try:
                r = requests.post(f"{BREEZE}/v1/audio/speech", files=files, timeout=900)
            except requests.RequestException as exc:
                raise gr.Error(f"The Breeze API at {BREEZE} couldn't be reached: {exc}")
            if r.status_code == 409:  # busy with a reader's request: wait and retry
                time.sleep(1.5)
                continue
            if not r.ok:
                try:
                    detail = r.json().get("detail")
                except ValueError:
                    detail = r.text[:300]
                raise gr.Error(f"Breeze answered {r.status_code}: {detail}")
            return r.content, int(r.headers.get("X-Sample-Rate", 24000)), time.time() - t0
    raise gr.Error("Breeze stayed busy; try again in a moment.")


def reader_text(t: str) -> str:
    """What the reader does to text before sending it: Breeze performs "(sigh)" and "[laugh]"."""
    t = re.sub(r"\s*[(\[]\s*", " — ", t)
    t = re.sub(r"\s*[)\]]\s*", " — ", t)
    t = re.sub(r"(\s—)+", " —", t)
    return re.sub(r"^\s*—\s*|\s*—\s*$", "", t).strip()


def reader_spans(text: str) -> list[str]:
    """Paragraphs, split at sentence ends when longer than the reader's span limit."""
    spans = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        cur = ""
        for sentence in re.findall(r"[^.!?…]+[.!?…]+[\"'”’)\]]*\s*|[^.!?…]+$", para):
            if cur and len(cur) + len(sentence) > SPAN_MAX:
                spans.append(cur.strip())
                cur = ""
            cur += sentence
        if cur.strip():
            spans.append(cur.strip())
    return spans


def timing(audio_s: float, gen_s: float) -> str:
    return f"{audio_s:.1f} s of audio in {gen_s:.1f} s, **{audio_s / gen_s:.2f}× real time**" if gen_s > 0 else ""


# ---------- tab actions ----------

def try_text(text, voice_id, description, cfg, seed, like_reader):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type or paste some text first.")
    voice = find_voice(voice_id) if voice_id and voice_id != NO_VOICE else None
    if not voice and not (description or "").strip():
        raise gr.Error("Choose a saved voice, or describe a voice.")
    args = {}
    if voice:
        if voice.get("ref_audio"):
            args = {"ref_audio": VOICES / voice["ref_audio"], "ref_text": voice["ref_text"]}
        else:
            args = {"instruction": voice.get("instruction")}
    else:
        args = {"instruction": description.strip()}
    pieces = reader_spans(text) if like_reader else [re.sub(r"\s+", " ", text)]
    pcm_all, rate, gen_total, lines = b"", 24000, 0.0, []
    for i, piece in enumerate(pieces, 1):
        sent = reader_text(piece) if like_reader else piece
        pcm, rate, gen = breeze(sent, cfg, seed, **args)
        pcm_all += pcm + (b"\0\0" * int(rate * 0.35) if i < len(pieces) else b"")
        gen_total += gen
        if len(pieces) > 1:
            lines.append(f"{i}. {len(sent)} characters: {timing(len(pcm) / 2 / rate, gen)}")
    out = write_wav(pcm_all, rate, scratch())
    summary = timing(len(pcm_all) / 2 / rate, gen_total)
    keeps_up = "keeps up with reading" if gen_total < len(pcm_all) / 2 / rate else "slower than reading: the reader will pause"
    return str(out), "\n".join([f"{summary} ({keeps_up})"] + ([""] + lines if lines else []))


def design(description, sample, cfg, seed):
    description, sample = (description or "").strip(), (sample or "").strip() or SAMPLE_TEXT
    if not description:
        raise gr.Error("Describe the voice first.")
    pcm, rate, gen = breeze(sample, cfg, seed, instruction=description)
    out = write_wav(pcm, rate, scratch())
    return str(out), str(out), f"Seed {int(seed)}: {timing(len(pcm) / 2 / rate, gen)}. Listen; try another seed if it isn't right."


def save_design(clip, description, sample, seed, vid, name):
    if not clip:
        raise gr.Error("Generate a voice first.")
    vid, name = check_id(vid, name)
    shutil.copyfile(clip, VOICES / f"{vid}.wav")
    msg = add_voice({"id": vid, "name": name, "ref_audio": f"{vid}.wav", "ref_text": (sample or "").strip() or SAMPLE_TEXT,
                     "seed": int(seed), "cfg_scale": 1.0, "description": (description or "").strip()})
    return msg, *refresh_all()


def test_recording(recording, transcript, text, seed):
    if not recording:
        raise gr.Error("Record or upload a clip first.")
    if not (transcript or "").strip():
        raise gr.Error("Type the exact words spoken in the clip.")
    ref = to_wav(recording, scratch())
    pcm, rate, gen = breeze(reader_text((text or "").strip() or SAMPLE_TEXT), 1.0, seed, ref_audio=ref, ref_text=transcript.strip())
    out = write_wav(pcm, rate, scratch())
    return str(out), timing(len(pcm) / 2 / rate, gen)


def save_recording(recording, transcript, seed, vid, name):
    if not recording:
        raise gr.Error("Record or upload a clip first.")
    if not (transcript or "").strip():
        raise gr.Error("Type the exact words spoken in the clip: the voice needs its transcript.")
    vid, name = check_id(vid, name)
    to_wav(recording, VOICES / f"{vid}.wav")
    msg = add_voice({"id": vid, "name": name, "ref_audio": f"{vid}.wav", "ref_text": transcript.strip(),
                     "seed": int(seed), "cfg_scale": 1.0})
    return msg, *refresh_all()


# ---------- clips from a long recording ----------
# Each upload is kept in SOURCES/<id>/ until deleted: audio.wav (mono 24 kHz, the format voices use),
# meta.json (name, length, chapters) and, once transcribed, transcript.json (words with times, sentences).

def soft(n_outputs: int):
    """Show a refusal or failure as a warning and leave the outputs as they were. (A raised error covers the
    outputs with an error overlay, which on this tab would block clicks on the transcript and candidates.)"""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except gr.Error as exc:
                gr.Warning(exc.message, duration=12)
                return gr.skip() if n_outputs == 1 else (gr.skip(),) * n_outputs
        return inner
    return wrap


AUDIO_TYPES = [".mp3", ".m4a", ".m4b", ".aac", ".wav", ".flac", ".ogg", ".opus"]
CLIP_GOOD, CLIP_MAX, CLIP_MIN = 15.0, 20.0, 3.0  # seconds; longer clips leave too little room for paragraphs
SID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9-]*$")
CLEANUP = [("Even out loudness", "loudness"), ("Fade the edges", "fade"),
           ("Shorten long pauses", "pauses"), ("Reduce background noise", "noise")]
PICTURE_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}


def fmt_t(sec: float) -> str:
    sec = max(0.0, float(sec or 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m)}:{s:04.1f}"


def source_dir(sid: str) -> Path:
    if not sid or not SID_RE.match(sid) or not (SOURCES / sid / "audio.wav").exists():
        raise gr.Error("Choose a recording first.")
    return SOURCES / sid


def source_meta(sid: str) -> dict:
    return json.loads((source_dir(sid) / "meta.json").read_text("utf-8"))


_transcripts: dict = {}


def source_transcript(sid: str) -> dict | None:
    path = source_dir(sid) / "transcript.json"
    if not path.exists():
        return None
    key = (sid, path.stat().st_mtime)
    if key not in _transcripts:
        _transcripts.clear()
        _transcripts[key] = json.loads(path.read_text("utf-8"))
    return _transcripts[key]


def source_choices() -> list[tuple[str, str]]:
    out = []
    for d in sorted(SOURCES.iterdir(), reverse=True) if SOURCES.exists() else []:
        try:
            m = json.loads((d / "meta.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        done = " · transcribed" if (d / "transcript.json").exists() else ""
        out.append((f"{m['name']} · {fmt_t(m['duration'])}{done}", d.name))
    return out


def probe(path: str) -> dict:
    """Check the upload is audio (a cover picture is fine, a video isn't) and read its title and chapters."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_chapters", "-show_format", "-of", "json", path],
                       capture_output=True, timeout=120)
    if r.returncode:
        raise gr.Error("That file couldn't be read as audio.")
    info = json.loads(r.stdout or b"{}")
    streams = info.get("streams", [])
    if any(st.get("codec_type") == "video" and not st.get("disposition", {}).get("attached_pic")
           and st.get("codec_name") not in PICTURE_CODECS for st in streams):
        raise gr.Error("That file has video in it. Upload audio only: MP3, M4A, M4B, WAV, FLAC or OGG.")
    if not any(st.get("codec_type") == "audio" for st in streams):
        raise gr.Error("There's no audio in that file.")
    chapters = [{"start": float(c["start_time"]), "end": float(c["end_time"]),
                 "title": (c.get("tags") or {}).get("title") or f"Chapter {i + 1}"}
                for i, c in enumerate(info.get("chapters", []))]
    return {"title": ((info.get("format") or {}).get("tags") or {}).get("title", ""), "chapters": chapters}


def upload_source(file):
    """Store an upload; a refused one is explained and cleared from the upload box so another can be chosen."""
    try:
        return _upload_source(file)
    except gr.Error as exc:
        gr.Warning(exc.message, duration=12)
        if file:
            Path(file if isinstance(file, str) else file.name).unlink(missing_ok=True)
        return gr.skip(), None


def _upload_source(file):
    if not file:
        raise gr.Error("Choose an audio file.")
    path = Path(file if isinstance(file, str) else file.name)
    if path.suffix.lower() not in AUDIO_TYPES:
        raise gr.Error("Upload an audio file: " + ", ".join(AUDIO_TYPES) + ".")
    info = probe(str(path))
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40]
    sid = time.strftime("%Y%m%d-%H%M%S") + "-" + slug
    d = SOURCES / sid
    d.mkdir(parents=True)
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "24000",
                        "-c:a", "pcm_s16le", str(d / "audio.wav")], check=True, capture_output=True, timeout=3600)
        with wave.open(str(d / "audio.wav"), "rb") as w:
            duration = w.getnframes() / w.getframerate()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, wave.Error) as exc:
        shutil.rmtree(d, ignore_errors=True)
        detail = exc.stderr.decode(errors="replace")[:300] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise gr.Error(f"That audio couldn't be converted: {detail}")
    finally:
        path.unlink(missing_ok=True)  # Gradio's copy of the upload: only the converted audio is kept
    meta = {"id": sid, "name": path.name, "title": info["title"], "duration": duration, "uploaded": time.time(),
            "chapters": info["chapters"]}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return gr.Dropdown(choices=source_choices(), value=sid), None


@soft(5)
def transcribe(sid, progress=gr.Progress()):
    d = source_dir(sid)
    progress(0, desc="Sending the audio to the speech-to-text service…")
    try:
        with open(d / "audio.wav", "rb") as f:
            r = requests.post(f"{ASR}/transcribe", files={"file": ("audio.wav", f, "audio/wav")}, stream=True, timeout=(30, 3600))
        r.raise_for_status()
        result = None
        for raw in r.iter_lines():
            if not raw:
                continue
            msg = json.loads(raw)
            if "error" in msg:
                raise gr.Error(f"Speech-to-text failed: {msg['error']}")
            if "progress" in msg:
                progress(msg["progress"], desc=msg.get("status", "Transcribing…"))
            if "result" in msg:
                result = msg["result"]
    except requests.RequestException as exc:
        reason = "it took too long to answer" if isinstance(exc, requests.Timeout) else \
            f"it answered {exc.response.status_code}" if getattr(exc, "response", None) is not None else "no connection"
        raise gr.Error(f"The speech-to-text service at {ASR} couldn't be used ({reason}). Is the asr container running "
                       "(docker compose ps)? You can still choose a clip by its times and type the transcript.")
    if not result:
        raise gr.Error("Speech-to-text ended without a transcript.")
    (d / "transcript.json").write_text(json.dumps(result, ensure_ascii=False), "utf-8")
    return open_source(sid)


def chapter_choices(meta: dict) -> list[tuple[str, int]]:
    return [("Whole recording", -1)] + [(f"{c['title']} ({fmt_t(c['start'])})", i) for i, c in enumerate(meta["chapters"])]


def transcript_rows(sid: str, chapter: int) -> tuple[list[list], list[int]]:
    """Table rows for the sentences (in one chapter, or all), and which sentence each row is."""
    tr = source_transcript(sid)
    if not tr:
        return [], []
    segs = tr["segments"]
    keep = list(range(len(segs)))
    chapters = source_meta(sid)["chapters"]
    if chapter is not None and 0 <= chapter < len(chapters):
        c = chapters[chapter]
        keep = [i for i in keep if c["start"] <= segs[i]["start"] < c["end"]]
    return [[fmt_t(segs[i]["start"]), f"{segs[i]['end'] - segs[i]['start']:.1f} s", segs[i]["text"]] for i in keep], keep


def open_source(sid):
    """Show a recording: its details, chapters and transcript."""
    if not sid:
        return "", gr.Dropdown(choices=[], value=None, visible=False), [], [], gr.Button(visible=False)
    meta, tr = source_meta(sid), source_transcript(sid)
    lines = [f"**{meta['name']}** · {fmt_t(meta['duration'])}" + (f" · {len(meta['chapters'])} chapters" if meta["chapters"] else "")]
    lines.append(f"Transcribed: {len(tr['segments'])} sentences. Click one to select it." if tr else
                 "Not transcribed yet. **Transcribe** finds every sentence and its timing (about a minute for 30 "
                 "minutes of audio on the GPU), or choose a clip by its start and end times below.")
    rows, shown = transcript_rows(sid, -1)
    return ("\n\n".join(lines), gr.Dropdown(choices=chapter_choices(meta), value=-1, visible=bool(meta["chapters"])),
            rows, shown, gr.Button(visible=not tr))


def words_in(sid: str, start: float, end: float) -> list[dict]:
    tr = source_transcript(sid)
    return [w for w in tr["words"] if start <= (w["start"] + w["end"]) / 2 <= end] if tr else []


def snap(sid: str, start: float, end: float) -> tuple[float, float]:
    """Move clip edges that fall inside a word to just outside it, so no word is cut."""
    tr = source_transcript(sid)
    if not tr:
        return start, end
    words = tr["words"]
    for i, w in enumerate(words):
        prev_end = words[i - 1]["end"] if i else 0.0
        if w["start"] < start < w["end"] or (prev_end <= start <= w["start"] and w["start"] - start < 0.05):
            start = max(prev_end, w["start"] - min(0.12, (w["start"] - prev_end) / 2))
        nxt = words[i + 1]["start"] if i + 1 < len(words) else w["end"] + 1.0
        if w["start"] < end < w["end"] or (w["end"] <= end <= nxt and end - w["end"] < 0.05):
            end = min(nxt, w["end"] + min(0.2, (nxt - w["end"]) / 2))
    return round(max(0.0, start), 2), round(end, 2)


def sentence_span(sid: str, first: int, last: int) -> tuple[float, float]:
    segs = source_transcript(sid)["segments"]
    first, last = max(0, first), min(len(segs) - 1, last)
    return snap(sid, segs[first]["start"], segs[last]["end"])


def build_clip(sid: str, start, end, opts) -> tuple[Path, float]:
    """Cut [start, end] from the recording, apply the chosen clean-up, and return (WAV path, seconds)."""
    d = source_dir(sid)
    start, end = float(start or 0), float(end or 0)
    if end - start < 0.5:
        raise gr.Error("Choose a clip: the end has to be after the start.")
    with wave.open(str(d / "audio.wav"), "rb") as w:
        rate = w.getframerate()
        total = w.getnframes() / rate
        start, end = max(0.0, start), min(end, total)
        w.setpos(int(start * rate))
        pcm = w.readframes(int((end - start) * rate))
    opts = opts or []
    if "pauses" in opts:  # shorten silences between words longer than 0.6 s to 0.45 s
        words, cuts = words_in(sid, start, end), []
        for a, b in zip(words, words[1:]):
            gap = b["start"] - a["end"]
            if gap > 0.6:
                mid = (a["end"] + b["start"]) / 2 - start
                cuts.append((mid - (gap - 0.45) / 2, mid + (gap - 0.45) / 2))
        for a, b in reversed(cuts):
            pcm = pcm[:int(a * rate) * 2] + pcm[int(b * rate) * 2:]
    raw = write_wav(pcm, rate, scratch())
    dur = len(pcm) / 2 / rate
    filters = []
    if "noise" in opts:
        filters += ["highpass=f=70", "afftdn=nf=-25"]
    if "loudness" in opts:
        filters.append("loudnorm=I=-18:TP=-1.5:LRA=11")
    if "fade" in opts:
        filters += ["afade=t=in:d=0.03", f"afade=t=out:st={max(0.0, dur - 0.06):.3f}:d=0.06"]
    if not filters:
        return raw, dur
    out = scratch()
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-af", ",".join(filters), "-ac", "1", "-ar", str(rate),
                        "-c:a", "pcm_s16le", str(out)], check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise gr.Error(f"Clean-up failed: {exc.stderr.decode(errors='replace')[:300]}")
    return out, dur


def clip_status(dur: float) -> str:
    if dur > CLIP_MAX:
        return (f"**{dur:.1f} s: too long.** Keep it under {CLIP_MAX:.0f} s (5–15 s is best): the clip and its transcript "
                "share the server's prompt with each paragraph, and long clips make paragraphs fail.")
    if dur > CLIP_GOOD:
        return f"**{dur:.1f} s**: usable, but 5–15 s leaves more room for long paragraphs."
    if dur < CLIP_MIN:
        return f"**{dur:.1f} s: too short.** Aim for 5–15 s of speech."
    return f"**{dur:.1f} s**: a good length."


@soft(3)
def preview(sid, start, end, opts):
    """Rebuild the selected clip; returns (clip, transcript, status)."""
    if not sid or end is None or start is None or float(end) - float(start) < 0.5:
        return None, gr.skip(), ""
    clip, dur = build_clip(sid, start, end, opts)
    words = words_in(sid, float(start), float(end))
    text = " ".join(w["text"] for w in words) if words else gr.skip()
    return str(clip), text, clip_status(dur)


@soft(6)
def select_sentence(sid, shown, opts, evt: gr.SelectData):
    if not sid or not shown:
        return (gr.skip(),) * 6
    k = shown[evt.index[0]]
    start, end = sentence_span(sid, k, k)
    return start, end, [k, k], *preview(sid, start, end, opts)


@soft(6)
def change_span(sid, span, opts, d_first, d_last):
    """Add or drop a sentence at either end of the selection."""
    if not sid or not span or not source_transcript(sid):
        raise gr.Error("Select a sentence in the transcript first.")
    n = len(source_transcript(sid)["segments"])
    first, last = span[0] + d_first, span[1] + d_last
    first, last = max(0, min(first, n - 1)), max(0, min(last, n - 1))
    if first > last:
        raise gr.Error("That would leave nothing selected.")
    start, end = sentence_span(sid, first, last)
    return start, end, [first, last], *preview(sid, start, end, opts)


@soft(6)
def manual_times(sid, start, end, opts):
    """Times typed or nudged by hand: snapped to word gaps, then previewed."""
    if not sid or start is None or end is None:
        return gr.skip(), gr.skip(), None, gr.skip(), gr.skip(), ""
    s, e = snap(sid, float(start), float(end))
    return (s if s != float(start) else gr.skip()), (e if e != float(end) else gr.skip()), None, *preview(sid, s, e, opts)


@soft(4)
def pick_chapter(sid, chapter):
    rows, shown = transcript_rows(sid, chapter)
    if source_transcript(sid) or chapter is None or chapter < 0:
        return rows, shown, gr.skip(), gr.skip()
    c = source_meta(sid)["chapters"][chapter]  # not transcribed: start the selection at the chapter
    return rows, shown, round(c["start"], 2), round(min(c["end"], c["start"] + 10), 2)


def cand_rows(cands: list[dict]) -> list[list]:
    return [[c["n"], f"{fmt_t(c['start'])}–{fmt_t(c['end'])}", f"{c['dur']:.1f} s", c["seed"], c["speed"],
             c["transcript"][:80] + ("…" if len(c["transcript"]) > 80 else "")] for c in cands]


@soft(4)
def test_long(sid, start, end, opts, transcript, text, seed, cands):
    clip, dur = build_clip(sid, start, end, opts)
    if dur > CLIP_MAX:
        raise gr.Error(f"The clip is {dur:.1f} s; keep it under {CLIP_MAX:.0f} s (5–15 s is best).")
    if not (transcript or "").strip():
        raise gr.Error("The clip needs its transcript: transcribe the recording, or type exactly what is said.")
    pcm, rate, gen = breeze(reader_text((text or "").strip() or SAMPLE_TEXT), 1.0, seed, ref_audio=clip, ref_text=transcript.strip())
    out = write_wav(pcm, rate, scratch())
    audio_s = len(pcm) / 2 / rate
    cand = {"n": len(cands or []) + 1, "start": float(start), "end": float(end), "dur": dur, "opts": list(opts or []),
            "transcript": transcript.strip(), "seed": int(seed), "clip": str(clip), "result": str(out),
            "speed": f"{audio_s / gen:.2f}× real time" if gen > 0 else ""}
    cands = (cands or []) + [cand]
    return str(out), f"Candidate {cand['n']}: {timing(audio_s, gen)}", cands, cand_rows(cands)


def pick_candidate(cands, evt: gr.SelectData):
    c = cands[evt.index[0]]
    return c["start"], c["end"], c["opts"], c["transcript"], c["seed"], c["clip"], c["result"], clip_status(c["dur"]), None


@soft(4)
def save_long(sid, start, end, opts, transcript, seed, vid, name):
    clip, dur = build_clip(sid, start, end, opts)
    if dur > CLIP_MAX:
        raise gr.Error(f"The clip is {dur:.1f} s; keep it under {CLIP_MAX:.0f} s.")
    if not (transcript or "").strip():
        raise gr.Error("The clip needs its transcript.")
    vid, name = check_id(vid, name)
    shutil.copyfile(clip, VOICES / f"{vid}.wav")
    meta = source_meta(sid)
    msg = add_voice({"id": vid, "name": name, "ref_audio": f"{vid}.wav", "ref_text": transcript.strip(), "seed": int(seed),
                     "cfg_scale": 1.0, "source": f"{meta['name']} {fmt_t(start)}–{fmt_t(end)}"})
    return msg, *refresh_all()


def delete_source(sid, confirm):
    # Plain messages, not errors: an error would cover the confirmation checkbox.
    if not sid or not SID_RE.match(sid) or not (SOURCES / sid).is_dir():
        return "Choose a recording first.", gr.skip(), gr.skip()
    if not confirm:
        return "Tick **Yes, delete this recording** to delete it.", gr.skip(), gr.skip()
    name = source_meta(sid)["name"]
    shutil.rmtree(SOURCES / sid)
    _transcripts.clear()
    return f"Deleted {name}. Voices saved from it are kept.", gr.Dropdown(choices=source_choices(), value=None), False


# ---------- voice management ----------

def table() -> list[list]:
    return [[v["id"], v.get("name", ""), v.get("ref_text") or v.get("instruction", ""), v.get("seed", 42)] for v in load_voices()]


def refresh_all():
    """New values for every voice list in the studio."""
    choices = voice_choices()
    first = choices[0][1] if choices else None
    return (gr.Dropdown(choices=[(NO_VOICE, NO_VOICE)] + choices, value=first or NO_VOICE),
            gr.Dropdown(choices=choices, value=first),
            table())


def pick(vid):
    v = find_voice(vid) if vid else None
    if not v:
        return None, "", ""
    clip = VOICES / v["ref_audio"] if v.get("ref_audio") else None
    return (str(clip) if clip and clip.exists() else None), v.get("name", ""), v.get("ref_text") or v.get("instruction", "")


def rename(vid, name):
    voices = load_voices()
    for v in voices:
        if v["id"] == vid:
            v["name"] = (name or "").strip() or vid
    save_voices(voices)
    return f"Renamed {vid}.", *refresh_all()


def move(vid, step):
    voices = load_voices()
    i = next((k for k, v in enumerate(voices) if v["id"] == vid), None)
    if i is None:
        raise gr.Error("Choose a voice first.")
    j = max(0, min(len(voices) - 1, i + step))
    voices.insert(j, voices.pop(i))
    save_voices(voices)
    choices, voice_list, rows = refresh_all()
    return f"Moved {vid} to position {j + 1}.", choices, gr.Dropdown(choices=voice_choices(), value=vid), rows


def delete(vid, confirm):
    # Plain messages, not errors: an error would cover the confirmation checkbox.
    v = find_voice(vid)
    if not v:
        return "Choose a voice first.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    if not confirm:
        return f"Tick **Yes, delete it** to delete {vid}.", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    save_voices([x for x in load_voices() if x["id"] != vid])
    if v.get("ref_audio") and not any(x.get("ref_audio") == v["ref_audio"] for x in load_voices()):
        (VOICES / v["ref_audio"]).unlink(missing_ok=True)
    return f"Deleted {vid}.", *refresh_all(), False


# ---------- the interface ----------

with gr.Blocks(title="Voice studio") as studio:
    gr.Markdown("# Voice studio\nTry text with the Breeze TTS server, design or record voices, and manage the voices "
                "the reader offers as **Server voice**.")
    with gr.Tab("Try text"):
        with gr.Row():
            with gr.Column(scale=3):
                t_text = gr.Textbox(label="Text", value=TRY_TEXT, lines=10)
                t_reader = gr.Checkbox(label="Read it like the reader: one request per paragraph (up to ~600 characters), "
                                             "parentheses read aloud instead of performed", value=True)
            with gr.Column(scale=2):
                t_voice = gr.Dropdown(label="Voice", choices=[(NO_VOICE, NO_VOICE)] + voice_choices())
                t_desc = gr.Textbox(label="Description (used when no saved voice is chosen)",
                                    value="A warm, calm narrator with a clear, steady voice.", lines=2)
                t_cfg = gr.Slider(1.0, 6.0, value=1.0, step=0.5, label="cfg_scale (1 = the reader's setting; higher follows a description more closely)")
                t_seed = gr.Number(label="Seed", value=42, precision=0)
                t_go = gr.Button("Generate", variant="primary")
        t_audio = gr.Audio(label="Result", type="filepath", interactive=False)
        t_info = gr.Markdown()
        t_go.click(try_text, [t_text, t_voice, t_desc, t_cfg, t_seed, t_reader], [t_audio, t_info])
        # A saved voice is read with its own seed and cfg_scale, as the reader does.
        t_voice.change(lambda vid: ((v := find_voice(vid)) and (v.get("seed", 42), v.get("cfg_scale", 1.0))) or (gr.skip(), gr.skip()),
                       t_voice, [t_seed, t_cfg])

    with gr.Tab("Design a voice"):
        gr.Markdown("Describe a voice, generate a sample, and save the one you like. The sample becomes the voice's "
                    "reference clip, so every paragraph the reader asks for is spoken in that same voice.")
        with gr.Row():
            with gr.Column():
                d_desc = gr.Textbox(label="Description", lines=3,
                                    value="A warm, calm middle-aged woman with a clear, steady storytelling voice.")
                d_sample = gr.Textbox(label="Sample text (becomes the reference transcript)", value=SAMPLE_TEXT, lines=4)
                with gr.Row():
                    d_cfg = gr.Slider(1.0, 6.0, value=4.0, step=0.5, label="cfg_scale (4 recommended for descriptions)")
                    d_seed = gr.Number(label="Seed", value=42, precision=0)
                with gr.Row():
                    d_go = gr.Button("Generate", variant="primary")
                    d_next = gr.Button("Next seed")
            with gr.Column():
                d_audio = gr.Audio(label="Sample", type="filepath", interactive=False)
                d_info = gr.Markdown()
                d_clip = gr.State()
                with gr.Row():
                    d_id = gr.Textbox(label="Id", placeholder="narrator")
                    d_name = gr.Textbox(label="Name in the reader", placeholder="Warm narrator")
                d_save = gr.Button("Save as voice")
                d_msg = gr.Markdown()
        d_go.click(design, [d_desc, d_sample, d_cfg, d_seed], [d_audio, d_clip, d_info])
        d_next.click(lambda s: int(s or 0) + 1, d_seed, d_seed).then(design, [d_desc, d_sample, d_cfg, d_seed], [d_audio, d_clip, d_info])

    with gr.Tab("Record or upload"):
        gr.Markdown("Use 5–15 seconds of clean speech and type exactly what is said. Recording needs the studio to be "
                    "opened over HTTPS (or on localhost); uploading works anywhere. Trim the clip with the scissors button.")
        with gr.Row():
            with gr.Column():
                r_audio = gr.Audio(label="Reference clip", sources=["microphone", "upload"], type="filepath", editable=True)
                r_text = gr.Textbox(label="Exact transcript of the clip", lines=3)
                r_seed = gr.Number(label="Seed", value=42, precision=0)
            with gr.Column():
                r_try = gr.Textbox(label="Text to test with", value=SAMPLE_TEXT, lines=3)
                r_test = gr.Button("Test this voice", variant="primary")
                r_out = gr.Audio(label="Test", type="filepath", interactive=False)
                r_info = gr.Markdown()
                with gr.Row():
                    r_id = gr.Textbox(label="Id", placeholder="me")
                    r_name = gr.Textbox(label="Name in the reader", placeholder="My voice")
                r_save = gr.Button("Save as voice")
                r_msg = gr.Markdown()
        r_test.click(test_recording, [r_audio, r_text, r_try, r_seed], [r_out, r_info])

    with gr.Tab("From a long recording"):
        gr.Markdown("Upload a long recording of the voice (an audiobook chapter, a podcast: MP3, M4A, M4B, WAV, FLAC or "
                    "OGG), transcribe it, and click sentences to choose a 5–15 second clip. Test as many clips as you "
                    "like; each test is kept below as a candidate so you can compare them, then save the best as a voice. "
                    "Recordings stay here until you delete them.")
        l_span = gr.State()   # [first, last] sentence of a selection made from the transcript
        l_shown = gr.State([])  # which sentence each transcript row is
        l_cands = gr.State([])
        with gr.Row():
            with gr.Column(scale=5):
                l_upload = gr.File(label="Upload a recording", file_types=AUDIO_TYPES, type="filepath", height=110, elem_id="l-upload")
                l_source = gr.Dropdown(label="Recording", choices=source_choices(), value=None, elem_id="l-source")
                l_info = gr.Markdown(elem_id="l-info")
                l_transcribe = gr.Button("Transcribe", variant="primary", visible=False, elem_id="l-transcribe")
                l_chapter = gr.Dropdown(label="Chapter", choices=[], visible=False, elem_id="l-chapter")
                l_table = gr.Dataframe(headers=["start", "length", "sentence"], value=[], interactive=False, wrap=True,
                                       max_height=520, column_widths=["16%", "14%", "70%"], elem_id="l-table")
                with gr.Accordion("Delete this recording", open=False):
                    l_confirm = gr.Checkbox(label="Yes, delete this recording", value=False, elem_id="l-confirm")
                    l_delete = gr.Button("Delete recording", variant="stop", elem_id="l-delete")
                    l_dmsg = gr.Markdown(elem_id="l-dmsg")
            with gr.Column(scale=6):
                with gr.Row():
                    l_start = gr.Number(label="Clip start (seconds)", value=None, step=0.1, minimum=0, elem_id="l-start")
                    l_end = gr.Number(label="Clip end (seconds)", value=None, step=0.1, minimum=0, elem_id="l-end")
                with gr.Row():
                    l_add_prev = gr.Button("◀ Add sentence before", size="sm", elem_id="l-add-prev")
                    l_drop_first = gr.Button("Drop first sentence", size="sm", elem_id="l-drop-first")
                    l_drop_last = gr.Button("Drop last sentence", size="sm", elem_id="l-drop-last")
                    l_add_next = gr.Button("Add sentence after ▶", size="sm", elem_id="l-add-next")
                l_status = gr.Markdown(elem_id="l-status")
                l_opts = gr.CheckboxGroup(CLEANUP, value=["loudness", "fade"], label="Clean-up", elem_id="l-opts")
                l_clip = gr.Audio(label="Selected clip", type="filepath", interactive=False, elem_id="l-clip")
                l_text = gr.Textbox(label="Transcript of the clip (filled in from the transcription; it must match what is said)", lines=3, elem_id="l-text")
                with gr.Row():
                    l_try = gr.Textbox(label="Text to test with", value=SAMPLE_TEXT, lines=3, scale=4)
                    l_seed = gr.Number(label="Seed", value=42, precision=0, scale=1)
                l_test = gr.Button("Test this clip", variant="primary", elem_id="l-test")
                l_out = gr.Audio(label="Test result", type="filepath", interactive=False, elem_id="l-out")
                l_tinfo = gr.Markdown(elem_id="l-tinfo")
        gr.Markdown("### Candidates\nEvery clip you have tested, with the same text. Click one to load it again.")
        l_ctable = gr.Dataframe(headers=["#", "clip", "length", "seed", "speed", "transcript"], value=[], interactive=False, wrap=True, elem_id="l-ctable")
        with gr.Row():
            l_id = gr.Textbox(label="Id", placeholder="me", elem_id="l-id")
            l_name = gr.Textbox(label="Name in the reader", placeholder="My voice", elem_id="l-name")
            l_save = gr.Button("Save the selected clip as a voice", variant="primary", elem_id="l-save")
        l_msg = gr.Markdown(elem_id="l-msg")

        span_out = [l_start, l_end, l_span, l_clip, l_text, l_status]
        l_upload.upload(upload_source, l_upload, [l_source, l_upload])
        l_source.change(open_source, l_source, [l_info, l_chapter, l_table, l_shown, l_transcribe]) \
            .then(lambda: (None, None, None, None, "", "", [], []), None, [l_start, l_end, l_span, l_clip, l_text, l_status, l_cands, l_ctable])
        l_transcribe.click(transcribe, l_source, [l_info, l_chapter, l_table, l_shown, l_transcribe])
        l_chapter.change(pick_chapter, [l_source, l_chapter], [l_table, l_shown, l_start, l_end])
        l_table.select(select_sentence, [l_source, l_shown, l_opts], span_out)
        l_add_prev.click(lambda sid, sp, o: change_span(sid, sp, o, -1, 0), [l_source, l_span, l_opts], span_out)
        l_add_next.click(lambda sid, sp, o: change_span(sid, sp, o, 0, 1), [l_source, l_span, l_opts], span_out)
        l_drop_first.click(lambda sid, sp, o: change_span(sid, sp, o, 1, 0), [l_source, l_span, l_opts], span_out)
        l_drop_last.click(lambda sid, sp, o: change_span(sid, sp, o, 0, -1), [l_source, l_span, l_opts], span_out)
        for box in (l_start, l_end):
            box.input(manual_times, [l_source, l_start, l_end, l_opts], span_out, trigger_mode="always_last")
        l_opts.input(preview, [l_source, l_start, l_end, l_opts], [l_clip, l_text, l_status])
        l_test.click(test_long, [l_source, l_start, l_end, l_opts, l_text, l_try, l_seed, l_cands], [l_out, l_tinfo, l_cands, l_ctable])
        l_ctable.select(pick_candidate, l_cands, [l_start, l_end, l_opts, l_text, l_seed, l_clip, l_out, l_status, l_span])
        l_delete.click(delete_source, [l_source, l_confirm], [l_dmsg, l_source, l_confirm])

    with gr.Tab("Voices"):
        gr.Markdown("These are the voices the reader lists, in this order.")
        v_table = gr.Dataframe(headers=["id", "name", "transcript / description", "seed"], value=table(), interactive=False, wrap=True)
        with gr.Row():
            with gr.Column():
                v_pick = gr.Dropdown(label="Voice", choices=voice_choices())
                v_clip = gr.Audio(label="Reference clip", type="filepath", interactive=False)
                v_ref = gr.Textbox(label="Transcript", interactive=False, lines=3)
            with gr.Column():
                v_name = gr.Textbox(label="Name in the reader")
                v_rename = gr.Button("Rename")
                with gr.Row():
                    v_up = gr.Button("Move up")
                    v_down = gr.Button("Move down")
                v_confirm = gr.Checkbox(label="Yes, delete it", value=False)
                v_delete = gr.Button("Delete voice", variant="stop")
                v_msg = gr.Markdown()
                v_reload = gr.Button("Reload list")
        v_pick.change(pick, v_pick, [v_clip, v_name, v_ref])
        v_rename.click(rename, [v_pick, v_name], [v_msg, t_voice, v_pick, v_table])
        v_up.click(lambda v: move(v, -1), v_pick, [v_msg, t_voice, v_pick, v_table])
        v_down.click(lambda v: move(v, 1), v_pick, [v_msg, t_voice, v_pick, v_table])
        v_delete.click(delete, [v_pick, v_confirm], [v_msg, t_voice, v_pick, v_table, v_confirm])
        v_reload.click(refresh_all, None, [t_voice, v_pick, v_table])

    d_save.click(save_design, [d_clip, d_desc, d_sample, d_seed, d_id, d_name], [d_msg, t_voice, v_pick, v_table])
    r_save.click(save_recording, [r_audio, r_text, r_seed, r_id, r_name], [r_msg, t_voice, v_pick, v_table])
    l_save.click(save_long, [l_source, l_start, l_end, l_opts, l_text, l_seed, l_id, l_name], [l_msg, t_voice, v_pick, v_table])
    studio.load(refresh_all, None, [t_voice, v_pick, v_table])
    studio.load(lambda: gr.Dropdown(choices=source_choices()), None, l_source)

studio.queue(default_concurrency_limit=1)  # Breeze runs one request at a time


class Voices(StaticFiles):
    """The voices folder for the reader; never cached, so new and changed voices show up at once."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI()
app.mount("/voices", Voices(directory=VOICES), name="voices")


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "voices": len(load_voices())})


app = gr.mount_gradio_app(app, studio, path="/", root_path=ROOT_PATH or None, allowed_paths=[str(VOICES), str(WORK)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=True, forwarded_allow_ips="*")
