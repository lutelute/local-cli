#!/bin/bash
# Capture a demo GIF of the CLI in action using asciinema + agg.
#
# Requirements:
#   brew install asciinema agg
#
# Usage:
#   ./scripts/capture-demo.sh
#
# Output:
#   assets/demo.gif

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT="$PROJECT_DIR/assets/demo.gif"
CAST_FILE="/tmp/local-cli-demo.cast"

echo "Recording demo session..."
echo "  - Type naturally to show the CLI in action"
echo "  - Run: python -m local_cli"
echo "  - Try a prompt like: 'Create a hello.py that prints hello world'"
echo "  - Type /exit to quit"
echo ""
echo "Press Ctrl+D when done recording."
echo ""

asciinema rec "$CAST_FILE" \
  --cols 100 \
  --rows 30 \
  --title "Local CLI Demo"

echo "Converting to GIF..."
agg "$CAST_FILE" "$OUTPUT" \
  --cols 100 \
  --rows 30 \
  --font-size 14 \
  --theme monokai

echo "Done: $OUTPUT"
echo "Size: $(ls -lh "$OUTPUT" | awk '{print $5}')"
