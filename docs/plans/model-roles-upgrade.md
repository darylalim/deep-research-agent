# Upgrade plan: dependencies, models, evals and interface

**PR 1 (Phases 0 and 1) is merged to `main` and released as v0.4.0.** Start each later PR on its
own branch from `main`; PR 2's is `upgrade/pr-2`.

Status: PR 1 shipped as v0.4.0 on 2026-11-07. PR 2 is in progress on `upgrade/pr-2`; judge calibration and Phase 2 are done, and Phase 2b is next. Check the LangSmith trace quota first. Drafted 2026-11-06 against `main` at `c6c5392`, then revised after
`uv lock --upgrade` moved to `anthropic` 2.6.0, `langchain-anthropic` 1.7.10 and `deepagents`
0.7.34. Check `uv.lock` before starting, in case the `anthropic` decision below changed it.

## How to work through this plan

This will take several sessions. Work one PR at a time, and update *Progress* before ending a
session. The page order is not the run order: within PR 1, Phase 0's paid steps 6 and 7 run after
Phase 1, because they need the judge fix.

| PR | Phases | Needs | Ends with |
|---|---|---|---|
| 1 | Phase 0 (all but steps 6 and 7), Phase 1 (all but calibration), then Phase 0 steps 6 and 7 | Nothing | Release 0.4.0 |
| 2 | Phase 1 calibration first, then Phase 2 and Phase 2b | PR 1 merged | Evals ready for sweeps |
| 3 | Phase 3, then Phase 3b | PR 2 merged, sweep budget approved | Researcher decision recorded |
| 4 | Phase 4 | PR 3 merged | Release 0.4.1 |
| 5 | Phase 5 | PR 1 merged | Weekly smoke test |
| 6 | Phase 6 | All others merged | `CLAUDE.md` split |

Inside Phase 4, keep this order: theme, layout, screenshots, README, `.env.example`, description
and topics, then the release.

## What needs the user

Stop and ask at each of these. Nothing here is implied by an earlier approval.

| Where | Needed |
|---|---|
| Phase 0 step 6 | Paid: live tests and a real app turn |
| Phase 0 step 7 | Paid: abandoned-turn test |
| Phase 1 | Paid, pennies: one real judge call, the regrade, and the calibration run |
| Phase 1 calibration | The user labels 20 to 30 claims |
| Release 0.4.0 | Outward-facing: approve the merge, and the note added to the release body |
| Phase 2b | Paid: a short run with the new recorder |
| Phase 2b | The user labels 15 to 20 claim and source pairs, and decides if `citations_are_real` is a hard bar |
| Phase 3 | Paid: approve the budget after the baseline sweep shows real cost |
| Phase 3b | Paid: three runs of one example |
| Phase 4 theme | Choose serif or sans headings from a render |
| Phase 4 layout | Paid, only if the approval screen cannot be stubbed |
| Phase 4 layout | Paid: retake the screenshots |
| Phase 4 description and topics | Outward-facing: update the public repository page |
| Release 0.4.1 | Outward-facing: approve the merge |
| Phase 5 | Paid: first run, and agree to the ongoing weekly cost |

## Budget

These are rough estimates, not measurements, and they exist so the size is visible before work
starts. They scale the old 5-example sweeps ($1.83 to $4.49, older models) to 9 examples, and
`CLAUDE.md` warns that the real figure runs higher. Phase 3's baseline sweep replaces them.

| Item | Rough cost |
|---|---|
| Phase 0 live tests and app turns | A few dollars |
| Phase 1 judge calls, regrade and calibration | Under $1 |
| Phase 2b short run | A few dollars |
| Phase 3, nine sweeps | $30 to $75, possibly more |
| Phase 3b and screenshots | A few dollars |
| **One-off total** | **About $40 to $100** |
| Phase 5 weekly run | Unknown until measured, and ongoing |

## Progress

Tick items as they finish, and add what was measured beside them, so the next session does not
have to work it out again.

- [x] PR 1: Phase 0 offline work. Measured on 2026-11-06, except the rule check:
  - Rule check (*Tighten `.claude/settings.json`* step 4), 2026-11-07, fresh session in auto
    mode. With a dummy `.env.local`, `Read`, `cat`, `Write` and a shell redirect were all
    refused, so the user had to create and delete the file. `.env.example` could be read
    (Read and `head`) and edited (a line added and removed), so the `!` lines work. All three
    `ask` probes prompted: `cd … && uv run pytest -m live --collect-only -q`,
    `uv run pytest -q -m live --collect-only`, and
    `DEEP_RESEARCH_RESEARCHER_EFFORT=medium uv run python -m evals --help`. Recorded in `CLAUDE.md`.
  - Lowest versions: 218 passed on `streamlit` 1.67.0, `anthropic` 2.0.0,
    `langchain-anthropic` 1.7.7 and `deepagents` 0.7.8.
  - The 13 warnings on Python 3.14 are all one line, `langsmith/_internal/_beta_decorator.py:92`
    (`asyncio.iscoroutinefunction`, removed in 3.16). None come from this repo.
  - API version review: nothing to change beyond `JUDGE`. The client sends
    `anthropic-version: 2026-10-21` and `Authorization`. There is no `document` or `citations`
    block, no Admin API, no `MessageParam` call, no `betas` on Models and no `X-Api-Key`, in the
    repo or in `langchain_anthropic` and `deepagents`. The SDK is now Apache 2.0, not MIT.
  - Docs confirm the `!` lines, and that `ask` prompts in auto mode and matches past a leading
    env assignment, so the third `ask` pattern is redundant but harmless.
- [x] PR 1: Phase 1 judge and regrade. Steps 1 to 4, the regrade and the hand check are done.
  `effort` and the structured-output `format` merge correctly in `output_config` (checked
  offline). Correction to step 3: asserting on top-level keys misses a sampling parameter on
  `anthropic` 2.x, because `langchain_anthropic` moves it into `extra_body` and it is still
  sent. The new test merges `extra_body`, and so does the `test_config.py` guard, which had
  been unable to fail. Both were seen to fail with a parameter put back.
  Step 4, measured on 2026-11-06 with one real `claims_are_cited` grade (1,714 input and 805
  output tokens, 426 of them thinking, tracing off): the response was `['thinking', 'text']`,
  `parsed` was correct with `parsing_error` None, and `stop_reason` read `end_turn` from
  `raw.response_metadata`. The fixture had 4 cited and 2 uncited figures, and the judge scored
  4/6. The thinking block carried no text, because `display` is omitted. That is one short
  answer, so it proves the plumbing, not the judge's accuracy; the regrade and calibration cover that.
  Regrade tool: `evals/regrade.py`, read path checked against real LangSmith (free). Results go
  to `~/deep-research-regrades/`. Experiments to regrade: `judgesplit-full-8708ffe9` (2026-09-06,
  the only full judged sweep, 5 answers) plus `workflow-feef8724` and `workflow-d516e642`
  (2026-10-01, 1 answer each, the same Tavily and LangSmith question). All three ran on 2026-11-06:
  42 judge calls, three repeats each. Full grades are in `~/deep-research-regrades/`.
  - **Two of the September answers are failed runs, not research.** Tavily returned `Error 432`
    (quota) on every search, so the Tavily and SqliteSaver answers hold no sourced findings. An
    earlier note here blamed their `0/0 claims cited` on truncation. That was wrong, or at least
    unproven: the new judge also returned `0/0` on one repeat of the Tavily answer.
  - On the 5 normal answers: `claims_are_cited` mean 0.97 old, 0.92 new, so the new judge is
    stricter by about 0.05. `answers_the_question` was identical. Repeats moved a score by 0.07 at
    most and 0.04 on average, and `answers_the_question` did not move at all. Claim counts are
    less stable than scores (30 to 42 on one answer).
  - The failed runs: SqliteSaver cited 1.00 old, 0.44 new, and the true figure is 0, since it
    retrieved nothing and says so. New is closer and still wrong. `answers_the_question` went from
    0 to 1.00 on that answer, which fits the prompt's rule that a hedged answer counts. The Tavily
    answer cited 0.33, 1.00 and 0.50 across repeats, the only large swing (0.67).
  - **Defect found and fixed:** an answer with no claims scored 1.0 through
    `_coverage_score(0, 0)`, so a refusal could collect a perfect citation score. It is now a
    non-verdict. Not fixed: the same refusal still scores 0.33 or 0.50 when the judge finds a few
    claims, and SqliteSaver still reads as partly cited with no sources behind it.
  - **Hand check (step 3), 2026-11-07: Haiku 5.5 accepted.** All 5 normal reports were read
    against every claim either judge flagged; the reports came from LangSmith through
    `stored_answers`, read only. Old: 7 accusations, 2 true, 1 borderline, 4 false, 11 real
    gaps missed. New, as the union of 3 repeats: 19 accusations, 12 true, 3 borderline, 4
    false, 1 gap missed. Each of new's false accusations appeared in one repeat only, and two
    of the 15 gradings hold all four. Claim by claim:
    `~/deep-research-regrades/hand-check-2026-11-07.md`. The score shift is recorded in
    `CLAUDE.md`. A model did this check, so the user-labelled calibration in PR 2 still applies.
- [x] PR 1: Phase 0 steps 6 and 7 passed, 2026-11-07, on `anthropic` 2.6.0.
  - Step 6: `pytest -m live` passed 3/3 in 15s: caching with `cache_read > 0`, thinking replay
    after a tool result, and Tavily. A real app turn, run in a scratch state dir, reached a
    `write_file` approval; Approve resumed it to a cited answer, and the server log showed no
    errors.
  - Step 7: a second turn on the same thread ran two researchers and reached approval, then
    was abandoned. A new question then answered with no 400, and the checkpoint kept every
    thinking block's text. Recorded in `CLAUDE.md`. Account age was not recorded.
  - Found on the way, not fixed: **LangSmith is over its monthly trace limit.** Every trace
    upload returned 429 "Monthly unique traces usage limit exceeded". The app is unaffected,
    because tracing fails in the background, but `python -m evals --run` writes its
    experiment to LangSmith. Check this before Phase 2b's short run and Phase 3's sweeps.
    The app's tracing (on in `.env`) also spends the same quota.
  - Display quirk, not fixed: after an abandoned turn, its question and the next one render
    in a single user bubble, because the abandoned turn left no assistant text between them.
- [x] PR 1: Release 0.4.0 published and note added. PR #2 merged as `4aea065` on
  2026-11-07, and the tag points there. The release body opens with a *Compatibility*
  section: API version `2026-10-21`, the bearer-auth change, resolved versions, the
  `streamlit>=1.67` floor, and the judge-shift warning.
  - Follow-up, 2026-11-07: the account predates enforcement, so step 7 was re-checked
    with the opt-in check. Zero breaks across 34 captured requests; see `CLAUDE.md`. No
    harness change is needed.
- [x] PR 2: Judge calibrated, 2026-11-07. `evals/calibration.json` holds 25 claims and 5
  question parts from 6 stored reports, labelled blind by the user. `--calibrate` (3 repeats,
  30 Haiku calls) scored the citation judge at **55/75 (73%) blind: FAIL**. The output
  schema was working: 6 of the 8 disagreements recurred in every repeat with coherent
  reasons, and the split was over the rule. The user chose the strict reading (a citation
  covers only the unit it closes) and revised c13 to c16 to uncited, stored as `adjudicated`.
  On those, the judge scores **67/75 (89%): PASS**, at 22, 22 and 23 of 25 per repeat. Known
  bias: the judge is lenient on lead-in citations and on bullets under a table (c08 and c22),
  so sweep citation scores read about 2 claims in 25 high. The answer judge flagged p03 in
  all three repeats, so it scored 12/15. It has no bar, and with no part labelled unanswered
  it can show only false accusations. Run saved at
  `~/deep-research-regrades/calibration-20261107T191649Z.json`.
- [x] PR 2: Phase 2 researcher settings, 2026-11-07, offline. `config.Effort` and
  `parse_effort`, `RESEARCHER_MODEL` / `RESEARCHER_EFFORT`, and `build_model(*, model, effort)`.
  The researcher dict now carries its own `model`. Step 5 was read off the installed deepagents
  `graph.py`: `model` (`:717`) and `interrupt_on` (`:799`) resolve independently, and caching is
  appended unconditionally (`:755`). A spy on the built subagents confirmed it: the researcher's
  middleware stack matches the inheriting general-purpose subagent's, with `HumanInTheLoopMiddleware`
  over all four `GATED_TOOLS`. Added beyond the plan: an `Effort` vs `ChatAnthropic` parity test, a
  gate-and-cache test on the built researcher, and import-time env tests in a subprocess. All of
  them were verified red by breaking the source. Phase 3 note: a researcher on another model
  family gets that model's harness profile, found by identifier, so check its extras first.
- [ ] PR 2: Phase 2b metrics added and calibrated
- [ ] PR 3: `score=None` measured
- [ ] PR 3: Phase 3 sweeps run and decision recorded
- [ ] PR 3: Phase 3b memory example added
- [ ] PR 4: Theme, layout and tests
- [ ] PR 4: README, `.env.example`, description and topics
- [ ] PR 4: Release 0.4.1 published
- [ ] PR 5: Weekly smoke test running
- [ ] PR 6: `CLAUDE.md` split and checked

## Goal

| Role | Now | Target |
|---|---|---|
| Main agent | `claude-opus-5-5`, effort `high` | No change |
| Researcher subagent | Inherits the main agent's model and effort | Configurable on its own, then set from eval evidence |
| Eval judge | `claude-haiku-4-5-20251001`, `temperature=0`, `max_tokens=8192` | `claude-haiku-5-5`, effort `medium`, no `temperature`, higher `max_tokens` |

Also in scope: finishing the dependency upgrade to latest versions, including `anthropic` 2.x,
with project config tidying (Phase 0); theme, layout, README and `.env.example` updates (Phase 4); a scheduled
live smoke test (Phase 5); splitting `CLAUDE.md` into path-scoped rules (Phase 6); and new evals
for judge calibration (Phase 1), citation truth (Phase 2b) and memory reuse (Phase 3b).

Out of scope: adding server-side `fallbacks`, changing prompts, and changing the main agent's
model or effort.

## Rules for this work

- Read `CLAUDE.md` first, especially *When the INSTRUMENT is broken* and *Model constraint*.
- Work on a branch, not `main`. Commit or push only when asked.
- Change `__version__` in `deep_research/__init__.py` only at the two release steps, at the ends
  of Phases 1 and 4. CI publishes a release automatically when a version bump reaches `main`.
- Edit Python with `Edit`/`Write` so the PostToolUse hook runs. Add an import and its first use
  in the same edit, and leave the suite green after every step.
- **Do not run anything that spends money without the user's approval.** That covers
  `python -m evals --run` (including `--code-only`), `pytest -m live`, and the judge regrade.
- Fix the judge before changing the researcher. If both change together, a score movement
  cannot be attributed to either.

## Phase 0: finish the dependency upgrade

Decision: keep `anthropic` 2.6.0 rather than holding it below 2. Judge-graded evals stay broken
until Phase 1 lands, so **ship Phases 0 and 1 together**. Do not merge Phase 0 alone.

The upgrade is already committed on the `upgrade/phase-0` branch, and has not been pushed. It
contains these changes:

- `uv.lock`: the result of `uv lock --upgrade`.
- `deep_research/agent.py`, `deep_research/webui.py`, `tests/test_agent_wiring.py`: `Iterator`
  changed to `Generator` on three `@contextmanager` functions, because `ty` 0.0.99 fails on the
  old form.
- `tests/test_streamlit_page.py`: `page.expander` changed to `page.main.expander`, because from
  Streamlit 1.65 `AppTest.expander` includes the sidebar. Confirmed by bisecting: the test passes
  on 1.64.0 and fails on 1.65.0. The rescoped test was seen to fail when the work log was put
  back in an expander.

Offline state when handed over: `ruff` and `ty` clean, 218 tests passing on Python 3.11 and 3.14.

Steps:

1. Check out `upgrade/phase-0`, where these changes already are.
2. Run `uv sync --locked` and the checks under *Finish*, to confirm the state above still holds.
3. Add `anthropic` as a capped direct dependency, `"anthropic>=2,<3"`, in `pyproject.toml`, then
   regenerate the lock with `uv lock` (never hand-edit `uv.lock`).
   - Add a comment giving the reason. It arrived only through `langchain-anthropic`, which sets
     a floor and no cap, so a major version that changed the API version came in unannounced.
   - The repo also depends on it directly: `TestStopReasonsAreAccountedFor` reads its types.
   - The floor of 2 is deliberate. Below it, the SDK sends the old API version.
4. Identify the 13 pytest warnings on Python 3.14 (none were examined). Fix any that come from
   this repo, and note any that come from dependencies.
5. Finish the API version review. The offline suite cannot prove any of this:
   - Run `/claude-api api-version`, which reads `shared/api-version-migration.md`, and work
     through its `2026-10-21` checklist, including the SDK changes page it links for Python.
   - Confirm nothing here sends `temperature`, `top_p` or `top_k` other than `JUDGE`. Phase 1
     removes that one.
   - Confirm no request places a `document` block in an `assistant` message, or a `text` block
     with non-empty `citations` in a `user` message. This includes `tool_result` content.
   - Note that API keys now go out as `Authorization: Bearer`. This only matters if a proxy or
     gateway reads `X-Api-Key`.
6. **Paid. Ask first.** After Phase 1, run `uv run pytest -m live`, then one real app turn that
   reaches a `write_file` approval. Nothing has exercised the app against `anthropic` 2.6.0 yet.
   Confirm streaming, thinking replay after tool calls, the approval resume, and `cache_read > 0`.
7. **Paid. Ask first.** Test an abandoned turn against preserved thinking. This risk is
   unconfirmed, so establish whether it is real before changing anything.
   - Why: Opus 5.5 can reject replayed thinking blocks when earlier history was edited, and this
     is enforced for accounts created on or after 2026-08-31. `PatchToolCallsMiddleware`
     answers dangling tool calls by rewriting the whole thread, on exactly the turn after an
     abandoned approval.
   - Test: reach a `write_file` approval on a thinking turn, click "Abandon this turn", then send
     a new question on the same thread. Check whether it succeeds, returns a 400, or silently
     loses thinking. Also try it after a researcher has run.
   - If it fails: run `/claude-api preserved-thinking-migration` for the diagnosis. Stop and bring
     the options to the user, because the fix touches the page's state machine or the middleware.
   - If it passes: record the result in `CLAUDE.md`, and note the account's age, since an older
     account may not be enforced and a pass there proves less.
8. No action is needed on the approval gate unless
   `test_every_mutating_tool_the_model_is_offered_is_gated` goes red. It passes on `deepagents`
   0.7.34, so that upgrade added no unclassified tool.

Do not bump `__version__` here. The bump is the last step of Phase 1.

### If `anthropic` 2.x fails live

Step 7 covers the abandoned-turn risk. This covers any other failure in steps 6 and 7.

1. Find the cause first, and fix it if the fix is small and stays inside this plan.
2. If not, stop and bring the failure to the user, with this fallback as the recommended option:
   cap at `"anthropic>=1,<2"`, regenerate the lock with `uv lock`, and keep every other upgrade.
3. Phase 1 still applies under the fallback, because Haiku 5.5 rejects non-default sampling on any
   API version. Rerun the offline suite and steps 6 and 7 on the capped version, and rewrite the
   step 3 comment, since the reason for a floor of 2 no longer holds.
4. Record what failed in `CLAUDE.md`, so the next attempt starts from it. Do not ship 0.4.0 until
   the live checks pass on whichever version is chosen.

### Tidy `pyproject.toml`

Keep the file as it is apart from step 3's `anthropic` cap and these three fixes. Make them in the
same edit as step 3, so `uv lock` runs once.

1. Raise the Streamlit floor to `"streamlit>=1.67,<2.0"`.
   - Phase 4's layout uses `st.grid`, which is missing in 1.60.0, 1.62.0, 1.64.0 and 1.66.0 and
     present in the locked 1.67.0. So 1.67 is the real floor, whatever the older versions pass.
   - The old floor was also wrong by itself. Measured on 2026-11-06 with
     `--resolution lowest-direct`: on 1.57.0, 20 of the 218 offline tests fail, with
     `chat_input() got an unexpected keyword argument 'submit_mode'`.
   - Replace the comment on that line, which says 1.57 ships `st.chat_input(submit_mode=...)`, with
     the `st.grid` reason.
   - Confirm the new floor once, in a throwaway environment outside the repo:

     ```bash
     uv venv --python 3.11 "$SCRATCH/lowest"
     VIRTUAL_ENV="$SCRATCH/lowest" uv pip install --resolution lowest-direct --group dev -e .
     "$SCRATCH/lowest/bin/python" -m pytest -q -p no:cacheprovider
     ```

   - Update `CLAUDE.md`, which says `pyproject.toml` supports `streamlit>=1.57`. With a floor above
     1.62, Streamlit itself discards a queued value for a disabled widget, so the
     `if prompt and busy:` guard no longer has a supported version where it runs. Keep the guard
     and `test_a_delivered_question_is_refused_with_a_notice`, because it is cheap defence, but
     correct the stated reason in `CLAUDE.md`, the page's comment, and the test comments at
     `tests/test_streamlit_page.py:270` and `:313`. Do not remove it here.
2. Add `--strict-markers` to pytest's `addopts`, so the line reads
   `addopts = "-m 'not live' --strict-markers"`. Without it, a typo such as `@pytest.mark.lve`
   passes `-m 'not live'` and that test tries the real APIs. All 218 tests pass with the flag on.
   See it fail on a misspelled marker first, then restore the test.
3. Fix the comment above `[tool.hatch.version]`. It ends "then tag `v<that value>`", but CI has
   created the tag and release automatically since 2026-08-24.

Leave these alone:

- The other dependency floors. They pass the offline suite, which proves nothing about the live
  API, and `uv.lock` is what actually gets installed.
- `ruff>=0.16`. Ruff 0.16.0 still accepts every file as formatted, so the floor and comment hold.
- `requires-python`, the uncapped dev tools and the wheel settings, which are deliberate.
- Classifiers, `[project.urls]` and `[tool.ty]`. Nothing here publishes to PyPI, and `ty` needs no
  configuration.

Keep the lowest-version check as a one-off step. Do not add it to CI, because only `uv.lock` ships
here and a permanent job would mostly guard versions nobody installs.

### Add timeouts to `ci.yml`

Add `timeout-minutes: 10` to the `lint` and `test` jobs in `.github/workflows/ci.yml`. Runs take
about 40 seconds, but GitHub's default limit is 6 hours, so a hung test would hold up a release
for that long. Make no other changes to this file.

Leave these alone:

- `cancel-in-progress` on `main`. A cancelled release retries on the next green push.
- The 3.11 and 3.14 matrix, and the exact `setup-uv` pin. Both are deliberate and documented.
- Floating `actions/checkout@v7`. Do not pin actions to commit SHAs: the job with write access
  uses only this GitHub-maintained action, and each bump would need a hand-edited hash.
- Branch protection. `main` is unprotected, which is fine for one maintainer, because
  `needs: [lint, test]` already stops a red commit from being released.
- Live tests. They stay in `live.yml` (Phase 5), because they need secrets and cost money.

### Tighten `.gitignore`

Keep the hand-written file, and do not replace it with a template. Add:

```gitignore
# Secrets: any env variant, but keep the committed template
.env.*
!.env.example

# Build output (the project builds with hatchling)
dist/
build/

# macOS
.DS_Store
```

1. Keep `!.env.example` after `.env.*`, because later rules override earlier ones.
2. Update the comment on the `research-*.md` rule. It refers to `/export` and `cli.py`, which are
   gone. Reports now download through the browser, from `streamlit_app.py`. Keep the rule.
3. Check it:
   - `git check-ignore` matches `.env.local`, `.env.prod`, `.DS_Store`, `dist/x` and `build/x`.
   - `git check-ignore .env.example` matches nothing, and `git ls-files .env.example` still lists it.
   - `git ls-files -ci --exclude-standard` prints nothing, so no tracked file became ignored.

### Close the two hook gaps

Do this before Phase 1, which edits `evals/evaluators.py`.

**Gap 1: edits to `evals/` skip `pytest`.** `tests/test_evals.py` is the largest test file, but
`.claude/hooks/post-edit.sh` runs the suite only for `deep_research/`, `tests/` and
`streamlit_app.py`.

1. Add `evals/*` to the `case` line in step 3 of `post-edit.sh`.
2. Update the script's header comment, which says `evals/` is deliberately skipped.
3. Update the matching text in `CLAUDE.md`, in the PostToolUse hook bullet.
4. Check it: pipe a sample `{"tool_input":{"file_path":"<repo>/evals/evaluators.py"}}` into the
   script with `CLAUDE_PROJECT_DIR` set, while a test is deliberately failing. It should exit 2.

**Gap 2: `.py` files changed through Bash, and all `.md` files, skip the hook.** Add a `Stop`
hook. `ruff format` also formats Python code blocks inside Markdown, but `post-edit.sh` exits on
anything that is not `.py`. On 2026-10-02 this turned `main` red: CI run 36944106993 failed
`ruff format --check` on the snippet in `README.md`. Run 36939730705 failed the same lint job on
a README commit the day before, but its cause was not confirmed. Phase 4 edits that snippet.

1. Create `.claude/hooks/stop-check.sh`, and wire it in `.claude/settings.json` as
   `bash "${CLAUDE_PROJECT_DIR}"/.claude/hooks/stop-check.sh`, with `"timeout": 120`. Use the
   `bash <path>` form, as `CLAUDE.md` requires for every hook.
2. Behaviour:
   - Exit 0 straight away when `stop_hook_active` is true in the hook's JSON input. Without this,
     a failing check blocks every stop and the session can loop.
   - Exit 0 when `git status --porcelain` shows no changed or untracked `.py` or `.md` files.
   - Otherwise run `ruff format --check`, `ruff check`, `ty check` and `pytest -q`, in that
     order. Exit 2 with the failure on stderr.
   - Check only. No `--fix`, because a hook at the end of a turn must not rewrite files nobody
     will review.
3. Accepted cost: with uncommitted `.py` or `.md` changes, every turn ends with a run of several seconds,
   even a turn that changed nothing. Do not add caching unless this proves annoying.
4. Check it: run it with no `.py` or `.md` changes, with a passing change, with a failing change,
   with a badly formatted Python block in `README.md`, and with `stop_hook_active` true. Confirm
   the expected exit code each time.
5. Document it in the hooks bullet in `CLAUDE.md`, and update the bullet that says Bash changes
   are never checked: they are now checked at the end of the turn, not at the edit.

### Tighten `.claude/settings.json`

Keep the file. Apart from the `Stop` hook above, make only these two changes, in the same edit:

```jsonc
"permissions": {
  "deny": [
    "Read(.env)",
    "Edit(.env)",
    "Read(.env.*)",            // new
    "Edit(.env.*)",            // new
    "Read(!.env.example)",     // new: the template must stay readable
    "Edit(!.env.example)",     // new: Phase 4 edits it
    "Read(./.deep_research/**)",
    "Edit(./.deep_research/**)",
    "Edit(./uv.lock)"
  ],
  "ask": [                     // new
    "Bash(uv run pytest *-m live*)",
    "Bash(uv run python -m evals *)",
    "Bash(* uv run python -m evals *)"
  ],
  "allow": [ /* unchanged */ ]
}
```

The real file is JSON, so leave these comments out.

1. `.env.*` matches the `.gitignore` rule added above. The docs do not say `Read(.env)` covers
   `.env.local`, and their example lists `.env.*` separately. The `!` lines follow the docs'
   pattern for an exception inside one deny list.
2. The `ask` list enforces this plan's rule about asking before spending money. In a normal session
   these commands already prompt, because `allow` does not list them, so it only matters in more
   permissive modes. Check whether auto mode honours `ask`, and record the answer in `CLAUDE.md`.
   The third pattern is for Phase 3's commands, which start with environment variables. Confirm
   that each of these prompts: `uv run pytest -m live`, `uv run pytest -q -m live`, and a command
   starting `DEEP_RESEARCH_RESEARCHER_EFFORT=medium uv run python -m evals`. If a form does not
   match, record that and rely on the plan's rule to ask. All paid eval work goes through
   `python -m evals`, including the new `--regrade` and `--calibrate`, so one rule covers it.
3. Correct `CLAUDE.md`, which calls the allow list "the read-only `uv` invocations". `uv sync`,
   `uv lock` and `ruff format` all change files, and `ruff check --fix` passes the allow rule too.
4. Check the rules in a fresh session, because none of this has been tested. Create a throwaway
   `.env.local` with a dummy value, confirm that `Read` and `cat` on it are refused, confirm that
   `.env.example` can still be read and edited, then delete `.env.local`. Do not read the real
   `.env`. If the `!` lines do not work, stop and pick a narrower rule, because Phase 4 needs
   `.env.example`.

Leave these alone:

- `Bash(uv run pytest)` as an exact match. Bash rules match command text, so widening it to
  `uv run pytest *` would let `uv run pytest -q -m live` through without matching the `ask` rule.
- The `Bash(... *)` form, `$schema`, the `Edit|Write` matcher and `settings.local.json`.
- The sandbox. Deny rules stop built-in tools and commands like `cat .env`, but not `grep -r` or a
  Python script opening the file. Only the sandbox enforces this at OS level, which is more setup
  than a single-developer project needs. Say this in `CLAUDE.md`, so deny rules are not read as a
  hard boundary.

Sources: <https://code.claude.com/docs/en/permissions.md> and
<https://code.claude.com/docs/en/settings.md>. Recheck both before starting.

Hook changes may not apply to the running session. If a hook does not fire, check `/hooks` or
restart the session before assuming the script is wrong.

## Phase 1: move the judge to Haiku 5.5

Why now: `anthropic` 2.x sends API version `2026-10-21`, which rejects `temperature` on every
model, so `JUDGE` as written fails with a 400 on any judge-graded run. The offline suite cannot
see this, because its tests replace `JUDGE` with `_StubJudge`. Haiku 5.5 rejects `temperature=0`
on any API version too. If `anthropic` is held below 2, this phase is still needed before that
hold can be lifted.

1. In `evals/evaluators.py:115`, set:

   ```python
   JUDGE = ChatAnthropic(model="claude-haiku-5-5", effort="medium", max_tokens=16000)
   ```

   - Do not pass `temperature`, `top_p` or `top_k`.
   - Thinking is on by default and counts toward `max_tokens`, which is why the limit rises.
   - Stay at or below 21,333, or set `streaming=True`. `JUDGE` does not stream, and CLAUDE.md
     explains why a larger non-streaming request can hang rather than fail.
   - Do not disable thinking. Use effort as the control.
2. Rewrite the comments above `JUDGE` (`evals/evaluators.py:97-114`). They justify Haiku with
   `temperature=0` and the old limit.
3. Add a test to `tests/test_evals.py` that builds a request with
   `JUDGE._get_request_payload(...)` and asserts it has no sampling parameter, the expected
   model, and the expected effort. Nothing pins this today: the existing tests replace `JUDGE`
   with `_StubJudge`. Confirm the test bites by putting `temperature=0` back and watching it fail.
4. **Paid, pennies. Ask first.** Make one real judge call and confirm that:
   - `with_structured_output(schema, method="json_schema", include_raw=True)` parses correctly
     when the response contains thinking blocks. This is unverified.
   - `_graded` still reads `stop_reason` from `raw.response_metadata`.

### Check the new judge against the old one

No regrade tool exists in `evals/`, so add one as `python -m evals --regrade <experiment>`.
`evals/__main__.py` rejects anything but `--upload` and `--run` today, so extend its parser. Grading
stored answers avoids re-running the agent.

Before grading anything:

- Confirm a recent full experiment still exists in LangSmith. If none does, stop and tell the user.
  The fallback is one paid sweep graded by the new judge, with no old scores to compare, relying on
  the hand check and calibration.
- Save that experiment's per-example scores to a file outside the repo. `evaluate_existing` in
  `langsmith` may add feedback onto the same experiment and mix new scores with the old ones.
  Prefer reading its outputs and grading locally, and write nothing back to it.

1. Take the stored answers from one recent full experiment. Use the Haiku 4.5 scores already
   stored with it as the old judge's grades. Do not re-grade with Haiku 4.5: at `temperature=0`
   it needs the old API version, which `anthropic` 2.x no longer sends.
2. Grade each answer with `claims_are_cited` and `answers_the_question` using the new judge three
   times, to measure run-to-run variation now that `temperature=0` is gone. If the stored scores
   came from an older grading schema, say so rather than treating them as like for like.
3. Hand-check a sample of claims where the two judges disagree. Past judge errors were found
   this way, and agreement between judges does not prove either is right.

Accept Haiku 5.5 if its disagreements with the hand check are no worse than Haiku 4.5's and its
run-to-run variation is small next to the differences the evals need to detect. Otherwise try
`claude-sonnet-5-5` with the same settings. Its non-default sampling is also a 400, so the
`temperature` removal stands either way.

**Scores from before and after this phase are not directly comparable.** Record the score shift
measured here in `CLAUDE.md`.

### Calibrate the judge against human labels

The regrade above compares Haiku 5.5 with Haiku 4.5, so it shows whether the judges agree, not
whether either is right. This lands as the first item in PR 2, and must pass before Phase 3.

1. Take 20 to 30 claims from stored answers, across good and weak reports, and include claims with
   inherited attribution (a source closing a bullet or the line under a table), which the judge has
   misread before.
2. **The user labels them.** Show each claim in its context and record cited or uncited, plus
   answered or unanswered for a few question parts. Do not show the judge's verdict first, and do
   not label them yourself, because a model labelling a model's work is what this check replaces.
3. Save them as `evals/calibration.json`. Each entry holds the question, the **full report**, the
   claim and the label, because inherited attribution depends on the whole report.
4. Add `python -m evals --calibrate` (paid). It must exercise the real path: run the real
   `claims_are_cited` on each full report, then count a claim as flagged when its text matches an
   entry in `uncited_claims` after normalising whitespace and case. Print agreement, the false
   accusations (flagged, but labelled cited), and any flagged entry that matched no labelled claim.
5. Preset rule: require at least 85% agreement before using the judge in Phase 3. With about 25
   labels this catches a badly wrong judge, not a small drift, so report the counts and not only a
   percentage. Earlier in this repo a judge was wrong on 9 of 12 accusations.
6. If it fails, change the output schema before the prompt wording. `CLAUDE.md` records that judge
   false positives are an output-schema problem. Stop and tell the user if that does not fix it.

### Release 0.4.0

This release marks the `anthropic` 2.x crossing, the new API version, the Streamlit floor and the
Haiku 5.5 judge. It is a minor bump, because a major upstream crossing is bigger than the routine
dependency bump in 0.3.2.

Do not bump until all of these have passed. CI runs only the offline suite, so it would publish a
release that breaks against the real API:

- Phase 0 steps 6 and 7: the paid live tests, the real app turn and the abandoned-turn test.
- Phase 1's live parse check and regrade. Judge calibration is not part of this release: it gates
  Phase 3, not the app, so it is the first item in PR 2.

Then:

1. Change `__version__` to `"0.4.0"` in `deep_research/__init__.py`, and nothing else, as the last
   commit on the branch. Do not tag by hand, and do not run `uv lock`, because `uv.lock` does not
   record this project's dynamic version.
2. Open a PR, so `--generate-notes` has a title to list. v0.3.3's commits went straight to `main`,
   and its notes are only a compare link.
3. **Outward-facing. Ask first.** Merging publishes the release straight away, with no further
   step. Confirm with the user before merging.
4. After it publishes, confirm with `gh release view v0.4.0`, then edit the body on GitHub to add a
   compatibility note: the resolved versions from `importlib.metadata` (not the pin ranges), the
   new Streamlit floor, and the API version. Pushing an annotated tag does not work, because the
   job writes generated notes over it.

If a release must be withdrawn, delete the tag as well as the release, or the next push recreates
it.

## Phase 2: make the researcher configurable, with no behaviour change

1. In `deep_research/config.py`:
   - Add `RESEARCHER_MODEL`, read from `DEEP_RESEARCH_RESEARCHER_MODEL`, defaulting to `MODEL_NAME`.
   - Add `RESEARCHER_EFFORT`, read from `DEEP_RESEARCH_RESEARCHER_EFFORT`, defaulting to `EFFORT`.
     `effort=` is a `Literal`, so a plain string from the environment fails `ty`, and this repo has
     no `ty: ignore`. Add an `Effort` type, and a small parser that returns it or raises on an
     unknown value, so a typo fails at import and not on the first request. Test the parser.
   - Give `build_model()` keyword arguments `model=MODEL_NAME` and `effort=EFFORT`. Keep every
     other setting, especially `streaming=True` and
     `thinking={"type": "adaptive", "display": "summarized"}`.
2. In `deep_research/subagents.py`, add this to the dict from `build_research_subagent()`:

   ```python
   "model": build_model(model=RESEARCHER_MODEL, effort=RESEARCHER_EFFORT),
   ```

   `deepagents` accepts a `BaseChatModel` here (`SubAgent.model` in
   `deepagents/middleware/subagents.py`).
3. In `tests/conftest.py`, pop `DEEP_RESEARCH_RESEARCHER_MODEL` and
   `DEEP_RESEARCH_RESEARCHER_EFFORT` beside `DEEP_RESEARCH_MAX_TOKENS`. Otherwise a line in a
   developer's `.env` fails the default test below, and the hook then blocks every edit.
4. Add tests:
   - `tests/test_config.py`: by default, the researcher's payload matches the main agent's
     model and effort.
   - `tests/test_config.py`: a non-default model and effort reach the payload, with no sampling
     parameter and the same `thinking` shape.
   - `tests/test_agent_wiring.py`: the built agent's researcher really has the configured
     `model`. Inspect what `create_deep_agent()` receives, not just the dict.
5. Confirm by reading the installed package, not by assuming, that a subagent with its own
   `model` still gets the prompt caching middleware and inherits `interrupt_on`. The second one
   is a safety property: `mutations_require_approval` depends on it.
6. Update the *Env overrides* list in `CLAUDE.md`. Both new values resolve at import time, like
   the others.

Merge point: Phases 1 and 2 can ship without any researcher sweep, since the defaults keep the
current behaviour.

## Phase 2b: grade whether citations are true

Why: the evals measure whether a claim has a citation, not whether it is true.
`response_cites_sources` only asks if the answer contains a URL, `claims_are_cited` only asks if a
claim has a citation nearby, and `TurnRecorder` keeps nothing the searches returned. A cheaper
researcher could invent a URL or misread a source and pass every metric in Phase 3's rule.

Do this after Phase 1 and before any Phase 3 sweep, so every arm is graded the same way. No step
here changes the agent or a prompt.

### Record what the searches returned

1. In `evals/harness.py`, make `TurnRecorder` keep the URL and text of every result from a
   successful `tavily_search`, across all namespaces, and add them to `actions()` under a new
   `search_results` key. Add the key to `_COMPLETED_RUN_KEYS` in `evals/evaluators.py` as well.
   That list is a hand-copy, and `test_the_completed_run_keys_match_what_the_recorder_emits` compares
   the two, so expect one red hook run between the two edits, and make them back to back. A search counts
   wherever it happened, like `searched_the_web`. Read the real result shape off the installed
   `langchain_tavily` and from a stored trace, not from memory.
2. Dedupe on tool-call ids, never `BaseMessage.id`, as `CLAUDE.md` requires, and leave out searches
   that errored.
3. Test it with real message types, including a resumed superstep that re-emits a search.

### Add `citations_are_real`

1. Add a code evaluator, in `CODE_EVALUATORS`: the share of URLs in the response that the searches
   returned, using `_coverage_score`, with the invented URLs in the comment.
2. Normalise before comparing: lower-case the host, drop `www.`, the fragment, a trailing slash and
   tracking parameters such as `utm_*`. Keep other query parameters, because they can identify the
   page. Test this, because a false accusation here reads as fabrication.
3. Call `_ungradable` first, as every evaluator does, and check `search_results`. It reads one key,
   and a row holding only `response` must not mark every URL as invented.
4. With no URLs in the response, return a non-verdict with `_not_measured`, not a vacuous 1.
   `response_cites_sources` already scores that case. This breaks
   `test_every_code_evaluator_reads_a_key_the_harness_emits`, which feeds every code evaluator
   `"some prose"` and rejects `None`. Change that fixture to prose containing a URL, which keeps the
   test's purpose for every evaluator, and add a separate test for the no-URL non-verdict.
5. **Paid. Ask first.** Stored runs have no `search_results`, so run the new recorder with
   `uv run python -m evals --run --limit 3 --prefix citations-check`. The first three examples
   include delegated research, and this run also supplies the text for calibration below.
6. Read any flagged URL by hand. An orchestrator may legitimately cite a primary source it knows;
   if that is common, report the rate and decide with the user before making it a hard bar.

### Add `claims_are_supported`

1. Add a judge evaluator, in `JUDGE_EVALUATORS`. Find cited claims in code, not with a model, so the
   list is repeatable: split the response into sentences and table rows, and keep those containing
   a URL. Sample about 5, seeded from the question, so a regrade picks the same claims. This misses
   claims covered by a citation elsewhere in a paragraph, so record that as a limit. Skip a claim
   whose URL has no retrieved text, because `citations_are_real` already covers it.
2. For each claim, give the judge the claim and the retrieved text for its cited URL, and require a
   verdict of `supported`, `contradicted` or `cannot_tell`, with a verbatim quote as evidence.
3. `cannot_tell` is not a failure, because search text is short and a claim missing from it is not
   false. Score supported over supported plus contradicted, and report the `cannot_tell` count. If
   nearly everything is `cannot_tell`, the metric is not measuring anything, so say so.
4. Reuse `_graded`, `include_raw=True` and the `stop_reason == "max_tokens"` non-verdict, so a
   truncated grade cannot be repaired and scored.
5. Calibrate it the same way as the judge in Phase 1: the user labels 15 to 20 claim and source
   pairs from the short run above, added to `evals/calibration.json` with the retrieved text, and
   graded by `--calibrate`, with the same 85% rule.

### Finish this phase

- Follow *When the INSTRUMENT is broken*: test what each evaluator returns when there is nothing to
  read, and see every guard fail by breaking the source.
- Update `CLAUDE.md`: the evaluator list, what `citations_are_real` cannot see (a real URL that does
  not support the claim), and what `claims_are_supported` cannot see (claims outside its sample).

## Phase 3: choose the researcher setting

**Paid. Get approval for the budget first.** The only measured sweeps used the old 5-example
dataset on older models ($1.83 to $4.49). The dataset now has 9 examples, including an expensive
one, so run one baseline sweep and use its cost to estimate the rest. Do not compare these
numbers with the 5-example figures.

### First: measure how LangSmith treats `score=None`

Do this before any sweep. The decision rule below rests on experiment means, and `CLAUDE.md`
records that nobody has measured whether LangSmith leaves a `score=None` non-verdict out of the
mean or counts it as 0. Haiku 5.5's thinking may also produce more non-verdicts than Haiku 4.5.

1. Create a tiny throwaway experiment with a stub target and a stub evaluator, no agent and no
   judge, where some rows return `1.0` and some return `score=None`. For example, rows of 1, 1
   and None: a mean of 1.0 means excluded, and 0.67 means counted as 0.
2. Read the aggregate from the same place the sweeps will be read from, and check the LangSmith
   UI as well, since the two could differ.
3. Delete the throwaway experiment, and keep it away from the real dataset.
4. Record the result in `CLAUDE.md`, replacing the "unmeasured" wording.
5. Whatever the result, count non-verdicts per metric for each arm in Phase 3. If they are
   counted as 0, compute the means from per-row scores with non-verdicts excluded.

### Run the arms

Arms, three full sweeps each, all with the Phase 1 judge:

| Prefix | `DEEP_RESEARCH_RESEARCHER_MODEL` | `DEEP_RESEARCH_RESEARCHER_EFFORT` |
|---|---|---|
| `researcher-opus-high` (baseline) | `claude-opus-5-5` | `high` |
| `researcher-opus-medium` | `claude-opus-5-5` | `medium` |
| `researcher-sonnet-medium` | `claude-sonnet-5-5` | `medium` |

```bash
uv run python -m evals --upload
DEEP_RESEARCH_RESEARCHER_MODEL=claude-opus-5-5 DEEP_RESEARCH_RESEARCHER_EFFORT=medium \
  uv run python -m evals --run --prefix researcher-opus-medium
```

Check the flags against `evals/__main__.py` before running. Read each experiment's cost with
`client.read_project(project_name=..., include_stats=True)`.

Fix the decision rule before looking at results. Adopt the cheapest arm where all of these hold:

- `claims_are_cited` and `answers_the_question` means are within the baseline's observed range,
  and non-verdict counts are no higher than the baseline's.
- `citations_are_real` is no worse than baseline, and `claims_are_supported` is within the
  baseline's observed range, with no more contradicted claims than the baseline.
- `delegates_breadth`, `searched_the_web`, `mutations_require_approval` and
  `turn_stopped_cleanly` are no worse than baseline.
- Median cost per sweep is lower, with the full range reported. Same-config sweeps have varied
  by 2.45x, so three sweeps show a clear saving but cannot prove a small one. Compare cost per
  finished report, because a cheaper researcher that causes retries or extra delegation is not
  cheaper.
- A person has read a few reports from that arm and is satisfied.

If no arm passes, keep the baseline. That is a valid result.

If an arm passes:

1. Change the `RESEARCHER_MODEL` / `RESEARCHER_EFFORT` defaults in `config.py`, and update the
   Phase 2 default test deliberately.
2. If the researcher's model now differs from the main agent's, update the two places that show
   a single model: the page caption (`streamlit_app.py:218`) and the export header
   (`deep_research/turns.py:362`).
3. Run `uv run pytest -m live tests/test_live.py::test_prompt_caching_actually_serves_the_prefix_from_cache`
   (paid; without `-m live` it is deselected), since
   CLAUDE.md requires this after any `build_model()` change.

## Phase 3b: grade memory reuse

Why: every example starts with empty memory, because `_reset_state()` drops both databases. So
`checks_memory_first` proves the agent looked, and nothing grades whether it uses what it finds,
which is the point of long-term memory.

Do this after Phase 3. It changes the dataset from 9 examples to 10, and sweeps of different sizes
must not be compared, so it must not land in the middle of the arms.

1. Add an optional `seed_memories` input in `evals/dataset.py`, a mapping of path to text, and one
   example whose answer is in a seeded note. Use a note that is plainly marked with its date, on a
   topic that does not change quickly, and that contains its source URLs.
2. Extend `dataset.drift()` and `sync()`. Both compare outputs only today, so a seed held in inputs
   would never upload and would be invisible to the pre-spend check. Test both directions.
3. In `harness.research()`, write the seed after `_reset_state()` and before the turn, through the
   real store under `MEMORY_NAMESPACE`. `CompositeBackend` strips the route prefix, so
   `/memories/x.md` is stored as `/x.md`. Pass the seed through the example's inputs, because
   `research()` cannot read outputs, and read the stored file format from the installed `deepagents`.
4. Make `TurnRecorder` record the paths of the orchestrator's `read_file` calls in a new `read_paths`
   key, because only tool names are kept today. Add it to `_COMPLETED_RUN_KEYS`, as in Phase 2b.
5. Add a `reuses_memory` code evaluator: for an example with a seed, the orchestrator read the note
   before its first research action. For other examples, return the same exempt score
   `plans_with_todos` gives when `expects_plan` is false, not a non-verdict, because
   `test_every_code_evaluator_reads_a_key_the_harness_emits` rejects `None`.
6. Stop the other metrics scoring a correct agent down on this example:
   - `searched_the_web` has no exemption, and answering from the note must not score 0. Add an
     `expects_search` column, where a missing value means a search is expected.
   - `citations_are_real` must count URLs in the seed as known. Have the harness add them to
     `search_results`, marked as coming from memory.
   - Set `expects_persist` to false and `max_delegations` to 0 for this example.
7. Run `--upload` (free), and confirm `drift()` sees the change. Then run this one example
   three times (**paid, ask first**). There is no flag for picking an example by name, so add one or
   use a throwaway dataset. Do not tune anything off that alone, since `CLAUDE.md` records how
   misleading a few runs are.
8. Update `CLAUDE.md`, including that the dataset is now 10 examples, so no total should be compared
   with the 9-example figures.

## Phase 4: interface and documentation

Update `CLAUDE.md`:

- *Model constraint*: subagents no longer simply inherit the model, and the `researcher` dict
  now has a `model` key.
- *The judge is Haiku, not the app's Opus* (line 1145): new model, why `temperature=0` is gone,
  and the measured run-to-run variation.
- *What a sweep costs*: add the 9-example figures, clearly separate from the 5-example ones.
- Record the Phase 3 results and the decision, including a decision to keep the baseline.
- Record the upgrade: `anthropic` 2.x sends API version `2026-10-21`, and it used to be only a
  transitive dependency with no cap, so a major version arrived through `uv lock --upgrade`. It is
  now a capped direct dependency.
- Near the existing notes on Streamlit 1.62 and 1.64, record that from 1.65 `AppTest.expander`
  includes the sidebar, so tests should scope to `page.main`.
- Update the deepagents and test-count figures in `CLAUDE.md` if they have drifted.
- Correct two claims measured wrong on 2026-09-05. Re-measure both on the upgraded dependencies
  rather than copying these figures:
  - *"the system+tools prefix is ~11.9k tokens"* (*Prompt caching*): measured then at about
    6.3k from the real `_get_request_payload` output, using non-Claude tokenizers. Measure it
    with `messages.count_tokens` this time (free). The figure dates from deepagents 0.6.x and
    also appears in `tests/test_live.py:38`.
  - *"nothing here or in deepagents' middleware stack summarizes or trims a thread"* (under
    `_SILENT_STOPS`): `_DeepAgentsSummarizationMiddleware` is in the built stack. It did not fire
    because its trigger is 85% of the model's `max_input_tokens`, about 850k on a 1M model. The
    conclusion held but the stated reason did not. Confirm that the Sonnet 5.5 and Haiku 5.5
    profiles behave the same way, since a model with no profile falls back to a much lower trigger.

### Refine the theme

Do this before the README work, because the README screenshots show the theme. Edit only the
`[theme*]` sections of `.streamlit/config.toml`, and keep `[server]`. This is a refinement, not a
redesign.

Keep these unchanged: every surface colour, `textColor`, `primaryColor`, `linkColor`,
`borderColor`, both sidebars, the radii and `linkUnderline`. The two modes are an ink-and-paper
swap: light is warm paper (hue 45) with cool ink text (hue 213), and dark reverses it (surfaces at
hue 218, text at hue 42). Do not warm the dark surfaces, because that breaks the swap. Measured on
2026-11-06, the modes also match in level: lowest text contrast 12.9 and 12.6, page to panel step
1.10 and 1.11, border step 1.36 and 1.37.

1. Replace the semantic colours, so each hue is identical in both modes and only lightness changes.
   Set each `*TextColor` to the same value as its `*Color`:

   | Colour | Hue | Light | Dark |
   |---|---|---|---|
   | `red` | 356 | `#aa252e` | `#f27981` |
   | `orange` | 30 | `#9d5308` | `#efa45a` |
   | `yellow` | 45 | `#806000` | `#e7c86c` |
   | `green` | 145 | `#1f7443` | `#7dc79c` |
   | `blue` | 217 | `#2358ac` | `#7da9f1` |
   | `violet` | 257 | `#6946c1` | `#b49cf0` |
   | `gray` | 217 | `#5d646f` | `#9aa1ac` |

   Why `*TextColor` matters: Streamlit derives it when unset, and `:orange[...]` then rendered as
   near-black brown in light mode. Why the hues moved: light red and orange were nearly identical
   (luminance ratio 1.03), and light green was 4.37:1 on the sidebar's secondary surface, despite
   the file's claim that every semantic colour clears 4.5:1.
2. Set `codeTextColor` to each mode's `textColor` (`#1f2328`, `#e7e3da`). Unset, Streamlit uses
   `greenTextColor` on a custom theme, and green means success here.
3. In `[theme]`, set `baseFontSize = 16`, and
   `headingFontSizes = ["32px", "25px", "20px", "17px", "16px", "15px"]`. Correct the comment that calls 15 "slightly
   larger than Streamlit's default", because the default is 16.
4. In `[theme]`, add `headingFont = "serif"`. It is the most subjective change, so show the user a
   render and drop it if they prefer sans. Do not use a web font, because this is a local app.
5. Update the comments in `config.toml`: the ink-and-paper swap, the hue rule, that contrast is
   measured on `*TextColor`, and the new figures. Delete figures that no longer hold.
6. Add `tests/test_theme.py`, which reads `.streamlit/config.toml` with `tomllib` and computes WCAG
   contrast. Nothing checks these numbers today, so they are only comments. Assert, in both modes:
   - Every `*TextColor` and `linkColor` clears 4.5:1 on every text surface, including all sidebar
     surfaces, and `textColor` clears 7:1.
   - Every `*TextColor` is set explicitly, and `codeTextColor` equals `textColor`.
   - Each semantic colour has the same hue in both modes, within 2 degrees for hex rounding.
   - White on `primaryColor` clears 4.5:1, and `primaryColor` clears 3:1 on the page and panel.
   - Alert text clears 4.5:1 on its tint. Streamlit blends the colour at alpha 0.1 in light and 0.2
     in dark; measured lowest values were 4.77 and 4.88. Name the alphas in a comment, since they
     are Streamlit's internals and could change.

   See each assertion fail by breaking a value first. Add the file to the test list in `CLAUDE.md`.
7. Use neutral avatars. The default user avatar is red, which means failure everywhere else in
   this theme, and the default assistant avatar is orange, which means a researcher's stop. The
   user decided this on 2026-11-06, so do not ask again.
   - In `streamlit_app.py`, add one constant per role, `USER_AVATAR = ":material/person:"` and
     `ASSISTANT_AVATAR = ":material/travel_explore:"`, in the same edit as their first use.
   - Pass `avatar=` to all four `st.chat_message` calls: the transcript loop at line 223 (choose by
     `_kind`), the question at 284, the approval panel at 322, and the streaming turn at 445. Find
     them with grep, since earlier phases may move these lines. A missed call would mix styles in
     one conversation.
   - Add a test in `tests/test_streamlit_page.py` that drives the transcript, a live turn and an
     approval, and asserts every chat message has the expected avatar. See it fail by removing
     `avatar=` from one call. Keep the existing page tests green.
8. Check it in a real render of both modes, not only the numbers. The preview used Playwright with
   the system Chrome, run outside the project (`uv run --no-project --with playwright`), with
   `color_scheme` set per mode. Plain `chrome --headless --screenshot` was unreliable. Check that a
   port is free first, because other Streamlit apps may be running, and always pass
   `--server.address localhost`. Check the feed lines, all three alert types, `st.code`, the
   approval controls, a report table and the sidebar.
9. Record in `CLAUDE.md`, next to the theme note under *Extending it*, why `*TextColor` is set, the
   hue rule, and the accepted limit below.
Accepted limit: the selected "Approve" label in dark mode is about 3.8:1, below 4.5:1 for text.
Streamlit uses `primaryColor` for that label, and no colour can clear 4.5:1 against the page while
also keeping white button text at 4.5:1. The selected state also shows a border and tint. Do not
lighten the primary to chase this, because that fails the button label instead.

### Adapt the layout to different displays

Do this after the theme and before the screenshots. It needs the Streamlit floor from Phase 0.

Streamlit does not tell the server the display size (`st.context` has no viewport value), so the
layout has to adapt by itself. Do not add custom CSS. Measured on 2026-11-06 with a sample app of
the same structure, not the real page, because reaching the approval screen needs a paid turn.
Characters per line are estimates from average character width:

| Display | Width | Characters per line now | Problem now |
|---|---|---|---|
| Phone | 390 | 45 | Wide tables run past the screen edge |
| Tablet, or half a screen | 820 | 64 | Work log 157px wide, and "Abandon this turn" ends at 853px, off the screen |
| Laptop | 1366 | 126 | Long, but workable |
| Desktop | 1920 | 209 | Too long to read comfortably |
| QHD | 2560 | 304 | Same |
| Ultrawide | 3440 | 435 | Same |

The 820 case is the serious one: that button is the only way out of a stuck approval. `st.columns`
stacks only below 640px of viewport width, however little room the open sidebar leaves.

1. Add a `READING_WIDTH_PX = 880` constant in `streamlit_app.py`, in the same edit as its first use.
2. Create `main = st.container(width=READING_WIDTH_PX)` once, and draw the title, transcript, turn
   and approval through `with main:` blocks. Keep `layout="wide"`, and keep the sidebar outside it.
   **Keep `st.chat_input` at the top level, outside every `with` block.** Streamlit pins it to the
   bottom only when it has no layout container around it (`elements/widgets/chat.py`), and it sits
   at `streamlit_app.py:112`, between the title and the transcript, so wrapping everything would
   move it inline to the top of the page. Pass it `width=READING_WIDTH_PX`. Measured in a sample
   app, the two lined up exactly at every size, and text stayed at about 123 characters.
3. Stay left-aligned. Do not centre the column: the pinned chat input stays left, and at 3440px it
   sat 1,010px away from the transcript. Accept the empty space to the right on big monitors.
4. Check the structure as you go, because this touches the page's state machine.
   - `st.stop()` must still work from inside the container.
   - The transcript's work-log expanders move under it, too.
   - `approval_panel` is a fragment that creates its own `st.chat_message`. Confirm it still
     renders and that both exits still escalate, with `test_submitting_a_decision_resumes_the_turn`.
   - The `st.status` block must still stream into the same place.
5. In `approval_panel`, replace `st.columns([2, 3], gap="medium")` with
   `st.grid(2, min_column_width=340, gap="medium")`, and take the two areas with `grid.cell()`. A
   grid responds to its own width, so at 820 it stacks and the button stays on screen. On wide
   displays the log and form are equal at 396px each. Keep the `else` branch: an empty log must
   still leave the form full width.
6. Lower `WORK_LOG_HEIGHT_PX` from 480 to 360. On a 390x844 phone, the log and decision buttons then
   span 802px, not 922px, so they fit in one screen.
7. Update `TestTheWorkLogSitsBesideTheDecision` in `tests/test_streamlit_page.py`, which reads
   `page.columns` in two tests. `AppTest` has a `grid` accessor, so check what it exposes and keep
   every existing assertion's meaning: the log is open, in a different area from the form, and an
   empty log does not split the screen. See each one fail by breaking the source.
8. Add a test that `st.chat_input` and the main container share `READING_WIDTH_PX`, so they cannot
   drift apart and put the input off-axis. `AppTest` exposes no width on the chat input, so stub
   `st.chat_input` to capture its arguments, as the existing stubbed test does, and read the
   container's `block.proto.width_config.pixel_width`. Also assert the input is still pinned, using
   its proto's `is_auto_positioned_at_bottom`, and see that fail by moving it inside a `with` block.
9. Check it on the real page in a browser, at 390, 820, 1366, 1920, 2560 and 3440. Use the method
   under *Refine the theme*, with a free port and `--server.address localhost`. Reaching an
   approval normally costs money, so seed the page state with a stub if the tests' approach allows
   it, and ask before running a paid turn. Confirm that:
   - No control ends past the right edge of the window, with the sidebar open.
   - The input and transcript share left and right edges at every width.
   - The approval split stacks at 820 and sits side by side at 1366.
10. Update the comments that this makes wrong: the `layout="wide"` note above `set_page_config`,
    which says a capped column was rejected, the "Below 640px the columns stack" comment in
    `approval_panel`, and the `WORK_LOG_HEIGHT_PX` note.
11. Record the result in `CLAUDE.md`, under *The app*, where it says a capped, centered reading
    column was rejected. Centring is still rejected, but `width=` makes a left-aligned cap work.
    Also note that `st.columns` stacks by viewport and `st.grid` by its own width.
12. **Paid. Ask first.** Retake the screenshots in `docs/images/`, which show the old theme and
    layout, now that both are final. Use a real research turn, or an existing thread if one is
    still available, and keep the same filenames so the README needs no edit.

Accepted limits:

- On a phone, wide Markdown tables still run past the edge. They do not scroll, and fixing that
  needs CSS. The app binds to `localhost`, so a phone cannot reach it; narrow windows are the real
  case, and they are wide enough for most tables.
- The approval preview is narrower than it is on a big monitor today. Its lines wrap and it
  scrolls, so nothing is hidden.
- 880 is a judgement between line length and table room. About 730 would give roughly 100
  characters per line, but that is the width that squeezed report tables before. Change the
  constant only with a real report in view.

### Update `README.md`

Keep the current structure and order. It answers a new reader's questions in the right sequence,
and the detailed reasoning stays in `CLAUDE.md`. Do this after Phase 3, because the cost figures
and the `How it works` snippet depend on its result. The sections become:

Intro and screenshot, Setup, Run, **Configuration** (new), **Cost** (new), How it works, Project
layout, Development, Extending it, License.

Add a **Configuration** section after Run:

- A small table of `DEEP_RESEARCH_MODEL`, `DEEP_RESEARCH_MAX_TOKENS`, `DEEP_RESEARCH_STATE_DIR`,
  and Phase 2's `DEEP_RESEARCH_RESEARCHER_MODEL` and `DEEP_RESEARCH_RESEARCHER_EFFORT`, each with
  its default and purpose. Read the defaults from `deep_research/config.py` rather than copying
  them from this plan.
- Say that these go in `.env` or the environment and are read once at startup, so restart the app
  after changing one.
- Put the backup note below in this section, as a *Data and backups* subsection, because
  `DEEP_RESEARCH_STATE_DIR` decides where the data lives.

Add a short **Cost** section after Configuration:

- Say that every question spends real Anthropic and Tavily usage, and that the cost of a turn
  varies widely with how much research it needs.
- Quote a range from the Phase 3 sweeps, not a single figure and not a price per question. The
  range in `CLAUDE.md` (47k to 1M tokens per example) comes from older models and the 5-example
  dataset, so do not reuse it.
- Keep it to a few lines, and name the Anthropic console as the place to check real spend.

Small follow-ups in existing sections:

- *How it works*: if Phase 3 changes the researcher's setting, update the `create_deep_agent`
  snippet and the two-layer table to match.
- *Development*: say that any `--run`, including `--limit 1` and `--code-only`, runs the real agent
  and costs money.
- Add a CI status badge under the title. Do not add other badges, a table of contents, a roadmap
  or a contributing section.

Backup note for the *Data and backups* subsection:

- `.deep_research/` holds conversations, pending approvals and long-term notes. It is gitignored,
  so git cannot restore it, and `memories.sqlite` is the only copy of the agent's notes.
- Say to stop the app before copying the folder. `SqliteSaver` uses WAL mode, so a copy taken
  while it runs can miss recent writes held in the `-wal` file.
- Mention that `DEEP_RESEARCH_STATE_DIR` moves this folder, so the backup path follows it.

Keep it to documentation. Do not add a backup command or script.

### Update `.env.example`

Keep it short and hand-written, with only the two required keys active. Target content:

```bash
# Copy to `.env` and fill in the two required keys. Loaded automatically.

# --- Required ---
# Claude: https://console.anthropic.com
ANTHROPIC_API_KEY=
# Web search: https://app.tavily.com (free tier available)
TAVILY_API_KEY=

# --- Optional: LangSmith tracing, and required for `python -m evals` ---
# Tracing sends your questions and the agent's answers to LangSmith.
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=
# LANGSMITH_PROJECT=deep-research-agent

# --- Optional: overrides ---
# Uncomment a line only to change it, and never leave one blank.
# Restart the app after changing any of these.
# DEEP_RESEARCH_MODEL=claude-opus-5-5
# DEEP_RESEARCH_MAX_TOKENS=64000
# DEEP_RESEARCH_STATE_DIR=.deep_research
```

Also add Phase 2's `DEEP_RESEARCH_RESEARCHER_MODEL` and `DEEP_RESEARCH_RESEARCHER_EFFORT` as
commented-out lines, and use the final defaults from `deep_research/config.py` for every override.

Why these changes:

- Tracing becomes opt-in. Today `cp .env.example .env` sets `LANGSMITH_TRACING=true` with an empty
  key, although the README lists LangSmith as optional. Once a key is added, research questions
  go to a third party, so that should be a deliberate choice.
- The overrides must stay commented out, not blank. `os.environ.get(name, default)` returns an
  empty string for a blank line in `.env`, so `DEEP_RESEARCH_MAX_TOKENS=` crashes on `int("")` at
  import and `DEEP_RESEARCH_STATE_DIR=` resolves to the repo root.
- The API key comment no longer names a model, so it cannot go stale.

Before finishing, check whether `python -m evals` still records its experiment with
`LANGSMITH_TRACING` unset. Read the installed `langsmith` source first, and use a paid
`--run --limit 1` only with approval. If tracing is needed, say so under the Optional heading, so
that a reader uncomments both lines.

Update the `LANGSMITH_API_KEY` row in the README's Setup table if the wording no longer matches.

### Update the project description and GitHub topics

Use one description everywhere, so the copies cannot drift:

> A Streamlit research agent built on Deep Agents and Claude: it searches the web with parallel
> subagents, writes cited reports, and asks before saving anything.

1. Set `description` in `pyproject.toml` to that sentence, on one line. Run `uv lock --check`
   afterwards, so CI's `uv sync --locked` cannot be surprised.
2. **Outward-facing. Ask first.** Merge the change, then update the public repository page:

   ```bash
   gh repo edit --description "<the sentence above>" \
     --add-topic deep-research --add-topic langsmith --add-topic web-search \
     --add-topic multi-agent --add-topic ai-agents
   ```

3. Confirm with `gh repo view --json description,repositoryTopics`. Expect 15 topics.

Keep the 10 existing topics. Public repository counts for the candidates on 2026-11-06:

| Topic | Repos | Decision |
|---|---|---|
| `deep-research` | 929 | Add |
| `langsmith` | 1,079 | Add |
| `web-search` | 1,679 | Add |
| `multi-agent` | 22,271 | Add |
| `ai-agents` | 134,265 | Add, rather than `ai-agent` (44,141) |
| `citations` | 1,482 | Skip |
| `deep-agents` | 118 | Skip, because `deepagents` (265) is already set |
| `llm` | 174,909 | Skip, too broad |
| `python` | 1,008,466 | Skip, too broad, and GitHub already shows the language |

Leave `homepageUrl` blank, because the app runs locally and has no site to link to.

### Release 0.4.1

Do this last in Phase 4. It covers the Phase 3 researcher default, if it changed, plus the theme
and layout, matching 0.3.1 (a default model change) and 0.3.3 (layout and theme). Phases 2, 2b and
3b get no release of their own, and Phases 5 and 6 ride along with a later one, because they do not
change what a user sees.

Release even if Phase 3 keeps the baseline, because the theme and layout are visible changes. The
user decided on 2026-11-06 that visible interface changes qualify.

Follow the same steps as *Release 0.4.0*: `__version__` to `"0.4.1"` as the last commit, a PR, and
ask before merging. Confirm the README screenshots were retaken first. A compatibility note is
needed only if a dependency moved again.

## Phase 5: scheduled live smoke test in CI

Why: the offline suite cannot see API-level breaks. The judge's `temperature` failure passed all
218 tests. A small scheduled run against the real APIs would catch this class of problem.

1. Add `.github/workflows/live.yml`, separate from `ci.yml`:
   - Triggers: weekly `schedule` plus `workflow_dispatch`. Do not run on `push` or
     `pull_request`, because pull requests would spend money and could expose secrets.
   - `permissions: contents: read`.
   - Same setup as `ci.yml`: `actions/checkout@v7`, `astral-sh/setup-uv@v10.0.1` (exact patch,
     see `CLAUDE.md`), `uv sync --locked`.
   - Run `uv run pytest -m live` with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` from secrets.
     `tests/conftest.py` uses `setdefault`, so real values take priority over its dummy keys.
   - Set `timeout-minutes`, so a hung request cannot run up a long job.
2. The user has to add the two repository secrets. Use dedicated keys with a low spend limit.
3. Keep it out of the `release` job's `needs`. A flaky external API must not block releases.
4. Add a live judge check to `tests/test_live.py`: one real grade through `JUDGE`, asserting a
   parsed score. Without it this workflow would not have caught the failure that motivated it,
   because the three existing live tests never call the judge. `JUDGE` is in `evals/`, so confirm
   the import works in the live suite, as `evals.harness` refuses to run at the repo root.
5. **Paid. Ask first.** Merge first, because `workflow_dispatch` needs the file on the default
   branch. Then trigger it once and record the cost per run, so the weekly cost is known.
6. Document the workflow in the *CI* section of `CLAUDE.md`. Note that GitHub disables scheduled
   workflows after 60 days without repository activity.

## Phase 6: split `CLAUDE.md` into path-scoped rules

Why: `CLAUDE.md` was 1,551 lines and about 113k characters on 2026-11-06, an estimated 28k tokens
loaded every session. The Claude Code docs set a target of under 200 lines per file and warn that
longer files reduce adherence. The content is worth keeping, so move it and delete nothing.

Do this last, as its own PR, after every other phase has finished editing `CLAUDE.md`. Moving text
while also changing it makes the diff unreviewable.

How loading works, from <https://code.claude.com/docs/en/memory.md> (recheck before starting):

- `@path` imports load at launch with the importing file, so they save nothing.
- `.claude/rules/*.md` with `paths:` frontmatter load only when Claude reads or edits a matching
  file. A rule with no `paths`, or with frontmatter that fails to parse, loads at launch.
- Nested `CLAUDE.md` files do not fit, because the topics cut across directories.

Target layout. Section sizes were measured on 2026-11-06, before the other phases added text:

| File | Loads when touching | Content moved | Size then |
|---|---|---|---|
| `CLAUDE.md` | Always | What this is, commands, editing rules, orchestrator model, extending it, one-line hard rules | Target about 200 lines |
| `.claude/rules/app.md` | `streamlit_app.py`, `deep_research/webui.py`, `deep_research/turns.py`, `tests/test_webui.py`, `tests/test_streamlit_page.py`, `tests/test_activity_feed.py`, `tests/test_turn_parsing.py` | *The app* and *The HITL resume loop* | About 38k chars |
| `.claude/rules/evals.md` | `evals/**`, `tests/test_evals.py` | The three eval sections | About 33k chars |
| `.claude/rules/model.md` | `deep_research/config.py`, `tests/test_config.py`, `tests/test_live.py` | *Model constraint* and *Prompt caching* | About 17k chars |
| `.claude/rules/agent.md` | `deep_research/agent.py`, `deep_research/subagents.py`, `tests/test_agent_wiring.py` | *Two-layer persistence*, *The `interrupt_on` dependency*, *Lifecycle* | About 9k chars |
| `.claude/rules/ci.md` | `.github/**`, `pyproject.toml` | The CI, release, `ruff` and `ty` detail from *Commands* | Part of about 12k chars |

Steps:

1. Record the before figures: `wc -lc CLAUDE.md`, and the memory files listed by `/context`.
2. Move each section word for word into its rule file, with `paths:` frontmatter. Do not rewrite or
   shorten anything during the move.
3. Keep a short **hard rules** list in `CLAUDE.md`, because a rule file is not loaded until Claude
   touches a matching file, so it can be missing while Claude plans. Include at least:
   - Never send `temperature`, `top_p` or `top_k` from `build_model()`.
   - Never pass `thinking={"type": "disabled"}`, and keep `display="summarized"`.
   - Never change `MEMORY_NAMESPACE`.
   - Never wire `LocalShellBackend`.
   - Never add a second prompt caching middleware.
   - The answer comes from the checkpoint, never from stream chunks.
   - Add a mutating tool only with its gate and a checkpointer.
   - Do not tune the prompt off a handful of eval runs, and check the instrument before the agent.
4. End `CLAUDE.md` with a short index of the rule files, saying which paths load each one, so a
   reader can open one while planning.
5. Check that nothing was lost. Write a throwaway script outside the repo that confirms every
   non-blank line of the old `CLAUDE.md` (from `git show`) appears in `CLAUDE.md` or a rule file,
   and review what it reports. Expect only headings and the sentences deliberately summarised.
6. Fix references. Run `grep -rn 'CLAUDE.md' deep_research evals tests streamlit_app.py pyproject.toml
   .github .claude README.md`, and repoint comments that name a section that moved, such as
   `evals/__main__.py:7`. Also update cross-references inside the moved text. Comment-only edits
   still trigger the hook, so keep the suite green.
7. Check that each rule fires, because a typo in `paths:` fails silently. Start a fresh session,
   read one matching file per rule, and confirm with `/context` that the rule loads then and not at
   launch. Also confirm `CLAUDE.md` is near the 200-line target, and record the after figures in
   the PR.
8. Update the auto-memory notes in `~/.claude/projects/.../memory/` if they name moved sections.

Do not delete lessons or dedupe them against source comments here. An earlier measurement found
only about 10% of this file was safely deletable, and that work is separate from this move.

## Finish

```bash
uv run ruff format . && uv run ruff check --fix . && uv run ty check && uv run pytest
```

Every new guard must be seen to fail by breaking the source first, as `CLAUDE.md` requires.

## Sources to recheck

Prices and model rules came from the `claude-api` skill, whose model table was cached on
2026-10-06. Recheck prices before budgeting. Per million input/output tokens: Opus 5.5 $4/$20,
Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 up to 100K-token prompts ($0.50/$2.50 above that),
Haiku 4.5 $1/$5.
