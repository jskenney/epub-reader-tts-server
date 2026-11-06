"""Voice studio for the EPUB reader's Server voice.

A Gradio app next to the Breeze TTS 2 API that lets you:
  - try any text with a saved voice or a description, with the API's parameters,
  - design voices from a description and save them,
  - record or upload a reference clip, test it, and save it as a voice,
  - preview, rename, reorder and delete voices.

It also serves the voices folder (voices.json plus the reference clips) at /voices/, which the reader
fetches through the website's /tts-voices/ proxy. Voices are stored in VOICES_DIR, a bind-mounted folder
on the host (see docker-compose.yml).

Environment:
  BREEZE_URL         the Breeze API (default http://breeze-tts:7860)
  VOICES_DIR         where voices are stored (default /data/voices)
  STUDIO_ROOT_PATH   the path the studio is served at behind a proxy, for example /tts-studio
                     ("" when opened directly on its port)
  STUDIO_PORT        default 7861
"""

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
MANIFEST = VOICES / "voices.json"
WORK = Path(tempfile.mkdtemp(prefix="studio-"))
VOICES.mkdir(parents=True, exist_ok=True)

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
    studio.load(refresh_all, None, [t_voice, v_pick, v_table])

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
