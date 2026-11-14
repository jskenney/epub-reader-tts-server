#!/bin/sh
# Sets up the voice server on an Apple Silicon Mac, without Docker (see "On a Mac" in the README).
#
#   1. Fetches Gust, Shoofio's MLX port of the Breeze TTS 2 server, at the commit below, into mac/gust,
#      and applies gust.patch so it answers like the Linux server (README, "On a Mac").
#   2. Creates the Python environments: mac/gust/.venv (Breeze on MLX) and mac/.venv (the studio and
#      speech-to-text, on PyTorch with the Mac's GPU).
#   3. Downloads Gust's real-time checkpoint (about 7 GB, once) into the Hugging Face cache.
#
# Needs: macOS 14 or later on Apple Silicon with 16 GB or more, and uv, ffmpeg and git
# (brew install uv ffmpeg). Safe to run again: it only redoes what is missing or out of date.
set -eu

GUST_URL="https://github.com/Shoofio/gust-breeze-tts-2-mlx-server.git"
GUST_COMMIT="da4265fcaf4e971e6629120fd07908f145282c64"

cd "$(dirname "$0")"
MAC=$(pwd)
ROOT=$(cd .. && pwd)

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "mac/setup.sh needs an Apple Silicon Mac (this machine: $(uname -s) $(uname -m))." >&2
    exit 1
fi
for tool in uv ffmpeg git; do
    command -v "$tool" >/dev/null || { echo "$tool is missing: brew install uv ffmpeg git" >&2; exit 1; }
done

echo "== Gust at $(printf %.7s "$GUST_COMMIT"), with gust.patch"
[ -d gust/.git ] || git init -q gust
if [ "$(git -C gust rev-parse -q --verify HEAD 2>/dev/null || true)" != "$GUST_COMMIT" ]; then
    git -C gust fetch -q --depth 1 "$GUST_URL" "$GUST_COMMIT"
    git -C gust checkout -q -f "$GUST_COMMIT"
fi
# Applied already? Otherwise start from the pinned commit's files and apply it.
if ! git -C gust apply --reverse --check "$MAC/gust.patch" 2>/dev/null; then
    git -C gust reset -q --hard "$GUST_COMMIT"
    git -C gust apply "$MAC/gust.patch"
fi

echo "== Python environment for Breeze (mac/gust/.venv)"
(
    cd gust
    [ -x .venv/bin/python ] || uv venv -q --python 3.12
    # Gust's own install line (its README): mlx-audio declares newer transformers than it pins.
    uv pip install -q -r requirements.txt --overrides requirements-mac-overrides.txt
)

echo "== Python environment for the studio and speech-to-text (mac/.venv)"
[ -x .venv/bin/python ] || uv venv -q --python 3.12
uv pip install -q --python .venv/bin/python -r "$ROOT/studio/requirements.txt" -r "$ROOT/asr/requirements.txt" \
    torch==2.9.1 numpy

echo "== Model"
# The checkpoint and revision Gust's launcher expects (read from it, so they're named in one place).
REPO=$(sed -n 's/^MIXED_REPO="\(.*\)"$/\1/p' gust/scripts/start_breeze_mac.sh)
REVISION=$(sed -n 's/^MIXED_REVISION="\(.*\)"$/\1/p' gust/scripts/start_breeze_mac.sh)
SNAPSHOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--$(printf '%s' "$REPO" | sed 's|/|--|')/snapshots/$REVISION"
if [ -f "$SNAPSHOT/model.safetensors.index.json" ]; then
    echo "$REPO is already downloaded."
else
    echo "Downloading $REPO (about 7 GB; research and non-commercial licence, as the Linux weights)…"
    uvx --from huggingface_hub hf download "$REPO" --revision "$REVISION"
fi

mkdir -p "$ROOT/tts-voices" "$ROOT/tts-sources" "$ROOT/asr-models" logs
echo
echo "Done. Start the server with: mac/start.sh   (or at every login: mac/install-service.sh)"
