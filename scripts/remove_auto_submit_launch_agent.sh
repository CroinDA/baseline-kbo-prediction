#!/bin/bash
set -euo pipefail

LABEL="com.croinda.kbo-auto-submit"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"

echo "removed: $PLIST_PATH"
