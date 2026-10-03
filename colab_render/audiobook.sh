#!/usr/bin/env bash
# Run orchestrate.py as the owner of this repository, where the Kaggle login,
# the rclone Google Drive remote and the book data live. An agent running as
# root (Hermes) calls this script; files never end up owned by root.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
owner="${AUDIOBOOK_USER:-$(stat -c %U "$here")}"
if [ "$(id -un)" != "$owner" ]; then
  exec sudo -u "$owner" -H bash -lc 'export PATH="$HOME/.local/bin:$PATH"; exec python3 "$0" "$@"' \
    "$here/orchestrate.py" "$@"
fi
export PATH="$HOME/.local/bin:$PATH"
exec python3 "$here/orchestrate.py" "$@"
