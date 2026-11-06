#!/usr/bin/env bash
# Add a voice for the EPUB reader's "Server voice" (Breeze TTS 2).
#
# Design a voice from a description (generated once, then reused as the reference clip):
#   ./make-voice.sh design narrator "Warm narrator" "A warm, calm middle-aged woman with a clear, steady storytelling voice."
#
# Or use your own recording (clean speech, 5-15 s) with its exact transcript:
#   ./make-voice.sh clip me "My voice" recording.wav "The exact words spoken in the recording."
#
# Voices are saved in VOICES_DIR, the voices folder the studio serves to the reader (./tts-voices next to
# docker-compose.yml by default): <id>.wav plus an entry in voices.json. The studio does the same with a UI.
# Every request then sends that clip and transcript, which keeps the voice the same across paragraphs.
# Environment:
#   VOICES_DIR   the voices folder (default ./tts-voices: run from the folder with docker-compose.yml)
#   BREEZE_URL   the Breeze API, for example http://breeze-host:7860 (needed for "design")
#   VOICE_SEED   default 42
set -euo pipefail
API=${BREEZE_URL:-}
SEED=${VOICE_SEED:-42}
DEST=${VOICES_DIR:-./tts-voices}
SAMPLE_TEXT="The old house stood at the end of the lane, its windows catching the last of the evening light. Inside, someone had left a book open on the table, as if they meant to come back to it."

usage() { sed -n '2,17p' "$0" >&2; exit 2; }
[ $# -ge 4 ] || usage
mode=$1 id=$2 name=$3
[[ "$id" =~ ^[a-z0-9_-]+$ ]] || { echo "The id may only use a-z, 0-9, - and _" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
mkdir -p "$DEST"

case "$mode" in
  design)
    description=$4; text=${5:-$SAMPLE_TEXT}
    [ -n "$API" ] || { echo "Set BREEZE_URL to the Breeze API, for example: BREEZE_URL=http://breeze-host:7860 $0 ..." >&2; exit 2; }
    command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    echo "Designing '$name' ..."
    curl -fsS -X POST "$API/v1/audio/speech" \
      -F "text=$text" -F "instruction=$description" -F "cfg_scale=4" -F "seed=$SEED" \
      -D "$tmp/headers" -o "$tmp/audio.pcm"
    rate=$(grep -i '^x-sample-rate:' "$tmp/headers" | tr -dc '0-9'); rate=${rate:-24000}
    python3 - "$tmp/audio.pcm" "$DEST/$id.wav" "$rate" <<'PY'
import sys, wave
pcm, out, rate = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = open(pcm, 'rb').read()
with wave.open(out, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(data)
print(f"Saved {out} ({len(data) / 2 / rate:.1f} s). Listen to it: if it isn't right, run again with another VOICE_SEED.")
PY
    ref_text=$text; extra=$description ;;
  clip)
    src=$4; ref_text=${5:?the transcript of the recording is required}
    [ -f "$src" ] || { echo "No such file: $src" >&2; exit 1; }
    case "$src" in *.wav|*.WAV) cp "$src" "$DEST/$id.wav" ;; *) command -v ffmpeg >/dev/null || { echo "Use a .wav file, or install ffmpeg to convert $src" >&2; exit 1; }; ffmpeg -loglevel error -y -i "$src" -ac 1 -ar 24000 "$DEST/$id.wav" ;; esac
    extra="" ;;
  *) usage ;;
esac

python3 - "$DEST/voices.json" "$id" "$name" "$id.wav" "$ref_text" "$SEED" "$extra" <<'PY'
import json, os, sys
path, vid, name, wav, ref_text, seed, description = sys.argv[1:8]
data = json.load(open(path)) if os.path.exists(path) else {"voices": []}
voices = [v for v in data.get("voices", []) if v.get("id") != vid]
entry = {"id": vid, "name": name, "ref_audio": wav, "ref_text": ref_text, "seed": int(seed), "cfg_scale": 1.0}
if description: entry["description"] = description
voices.append(entry)
data["voices"] = voices
json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)
print(f"Added '{name}' ({vid}) to {path}: {len(voices)} voice(s).")
PY
