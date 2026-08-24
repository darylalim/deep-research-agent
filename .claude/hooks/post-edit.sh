#!/usr/bin/env bash
# PostToolUse(Edit|Write) — keep the CI gate green as Claude edits Python.
#
# Three steps, in this order and in ONE script on purpose: `ruff check --fix`
# rewrites the file, so a checker racing it in a parallel hook could read a
# half-rewritten tree. Sequential is deterministic. Within that constraint the
# order is cheapest-first — 0.10s, 0.24s, 5.1s measured — so the common failure
# is reported in a fraction of a second instead of after the whole suite.
#
#   1. ruff  — format + autofix the edited file, then surface anything ruff could
#              NOT autofix (this repo selects E,F,I,UP,B,SIM,RUF,BLE).
#   2. ty    — CI's `lint` job runs `ty check` alongside ruff, so leaving it out
#              here meant a type error surfaced as a red GitHub Actions run rather
#              than next to the file that caused it. Project-wide (ty has no useful
#              single-file mode) and measured at 0.24s, so it runs on EVERY .py edit
#              — including `streamlit_app.py` and `evals/`, which CI type-checks but
#              the pytest step below deliberately skips. Note that a DEAD
#              `# ty: ignore` is itself a failure here (`unused-ignore-comment`),
#              which is exactly the kind of error that otherwise only shows up in CI.
#   3. pytest — for edits under deep_research/ or tests/, run the offline suite.
#               ~5s, needs no keys and no network: conftest.py sets dummy keys and
#               redirects DEEP_RESEARCH_STATE_DIR to a tempdir, and pyproject sets
#               `addopts = -m 'not live'` so the paid `live` tests are deselected.
#
# exit 2 = block and feed stderr back to Claude. PostToolUse cannot un-write the
# file, but it does force Claude to fix it before moving on.
#
# settings.json invokes this as `bash <path>`, not as a bare executable. The file
# IS committed mode 100755, but the exec bit is deliberately not load-bearing: if
# it were ever lost (a Windows checkout, a zip download, the file recreated without
# +x), a directly-invoked hook fails with "permission denied" and simply stops
# running — no error surfaced, no checks performed, no sign the safety net is gone.
# Going through `bash` means the worst case is the hook still working.

set -uo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  file=$(printf '%s' "$input" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
fi

# Only Python, only inside this project.
[[ -n "${CLAUDE_PROJECT_DIR:-}" ]] || exit 0
[[ "$file" == *.py ]] || exit 0
[[ "$file" == "$CLAUDE_PROJECT_DIR"/* ]] || exit 0
[[ -f "$file" ]] || exit 0

cd "$CLAUDE_PROJECT_DIR" || exit 0
rel=${file#"$CLAUDE_PROJECT_DIR"/}

# --- 1. ruff: format, autofix, then report the leftovers ----------------------
uv run ruff format -q -- "$file" >/dev/null 2>&1
uv run ruff check -q --fix -- "$file" >/dev/null 2>&1

if ! leftover=$(uv run ruff check -- "$file" 2>&1); then
  {
    echo "ruff found issues in ${rel} that it could not autofix."
    echo "The CI 'lint' job runs 'ruff check' and will fail on these. Fix them:"
    echo
    echo "$leftover"
  } >&2
  exit 2
fi

# --- 2. ty: every .py file, because CI type-checks every .py file -------------
if ! tyout=$(uv run ty check 2>&1); then
  {
    echo "ty reports type errors after your edit to ${rel}."
    echo "The CI 'lint' job runs 'uv run ty check' and will fail on these."
    echo "Note: a DEAD '# ty: ignore' is itself an error (unused-ignore-comment),"
    echo "so an upgrade that FIXES a false positive turns its suppression red."
    echo
    printf '%s\n' "$tyout" | tail -30
  } >&2
  exit 2
fi

# --- 3. pytest: only for source/test edits ------------------------------------
case "$rel" in
  deep_research/*|tests/*) ;;
  *) exit 0 ;;
esac

if ! out=$(uv run pytest -q 2>&1); then
  {
    echo "The offline test suite is RED after your edit to ${rel}."
    echo
    echo "These tests guard this repo's load-bearing invariants — the Opus 5"
    echo "no-sampling-params rule, the GATED_TOOLS human-approval gate, the"
    echo "/memories/ durable route and its store namespace, the HITL decision"
    echo "protocol, and the deepagents 0.7 backend contract. A failure here is"
    echo "usually a real regression, not a flaky test. Do not proceed until green."
    echo
    printf '%s\n' "$out" | tail -40
  } >&2
  exit 2
fi

exit 0
