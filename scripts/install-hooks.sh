#!/usr/bin/env bash
# Install the project git hooks into .git/hooks/.
set -euo pipefail

HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

cp scripts/pre-commit-hook.sh "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✔ pre-commit hook installed at $HOOKS_DIR/pre-commit"
echo "  Runs: black (format) → ruff (fix) → git add (auto-stage) → pytest"
