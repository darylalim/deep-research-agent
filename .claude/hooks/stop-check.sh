#!/usr/bin/env bash
# Stop — the end-of-turn gate for what the PostToolUse hook cannot see.
#
# post-edit.sh matches `Edit|Write` on `.py` files, so a `.py` file changed through
# Bash is never checked there, and neither is any `.md` file. The second gap is real:
# `ruff format` also formats Python blocks inside Markdown, and on 2026-10-02 a README
# snippet turned `main` red on `ruff format --check` (CI run 36944106993).
#
# Check only, so no `--fix`: a hook at the end of a turn must not rewrite files nobody
# will review. Same order as CI's lint job, then the offline suite.
#
# exit 2 = block the stop and feed stderr back to Claude. `stop_hook_active` means this
# stop already follows a block, so exit 0 then or a failing check loops the session.
#
# Accepted cost: with uncommitted `.py` or `.md` changes, every turn ends with a run of
# several seconds, even a turn that changed nothing.

set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
else
  active=$(printf '%s' "$input" | python3 -c \
    'import json,sys; print(str(json.load(sys.stdin).get("stop_hook_active",False)).lower())')
fi
[[ "$active" == "true" ]] && exit 0

[[ -n "${CLAUDE_PROJECT_DIR:-}" ]] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

# `-uall` lists files inside an untracked directory rather than just the directory.
git status --porcelain -uall | grep -qE '\.(py|md)"?$' || exit 0

run() {
  local out
  if ! out=$("$@" 2>&1); then
    {
      echo "Stop check failed: $*"
      echo "CI runs this too, so fix it before ending the turn."
      echo
      printf '%s\n' "$out" | tail -40
    } >&2
    exit 2
  fi
}

run uv run ruff format --check
run uv run ruff check
run uv run ty check
run uv run pytest -q

exit 0
