#!/bin/sh
# Runs mac/start.sh at every login and restarts it if it stops (a launchd agent, the Mac's equivalent of
# Docker's "restart: unless-stopped"). mac/install-service.sh --uninstall removes it.
# Settings go in mac/settings.env (see mac/start.sh); restart the agent after changing them:
#   launchctl kickstart -k gui/$(id -u)/local.epub-reader-tts-server
set -eu

cd "$(dirname "$0")"
MAC=$(pwd)
LABEL=local.epub-reader-tts-server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$PLIST"
    echo "Removed; the voice server no longer starts at login."
    exit 0
fi
[ -x gust/.venv/bin/python ] || { echo "Run mac/setup.sh first." >&2; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" logs
# (uv and ffmpeg live in Homebrew's or uv's own folder, which launchd's PATH doesn't include.)
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/sh</string><string>$MAC/start.sh</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$MAC/logs/start.log</string>
  <key>StandardErrorPath</key><string>$MAC/logs/start.log</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "Installed: the voice server starts now and at every login (logs in mac/logs/)."
