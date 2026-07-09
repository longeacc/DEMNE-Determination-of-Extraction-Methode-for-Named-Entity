#!/usr/bin/env bash
# Git pre-commit hook — format, fix, auto-stage, then test.
# Install: bash scripts/install-hooks.sh
set -euo pipefail

# ── 1. Collect staged Python files ──────────────────────────────────────────
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -n "$STAGED" ]; then
    echo "▶ black: formatting staged Python files..."
    # shellcheck disable=SC2086
    black --line-length=100 --quiet $STAGED

    echo "▶ ruff: fixing staged Python files..."
    # shellcheck disable=SC2086
    ruff check --fix --quiet $STAGED || {
        echo "✖ ruff found errors it cannot auto-fix. Please fix them and re-commit."
        exit 1
    }

    # Re-stage any files that black/ruff modified
    # shellcheck disable=SC2086
    git add $STAGED
    echo "✔ Formatting applied and files re-staged."
fi

# ── 2. Run CPU tests ─────────────────────────────────────────────────────────
echo "▶ pytest: running CPU-only tests..."
python -m pytest -m "not cuda and not rocm" --tb=short -q --no-header || {
    echo "✖ Tests failed. Please fix them and re-commit."
    exit 1
}

echo "✔ All checks passed — proceeding with commit."
