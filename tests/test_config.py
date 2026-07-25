"""Tests for `config.py`: the credential gate and the model-construction invariant."""

from __future__ import annotations

import pytest

from deep_research import config


class TestMissingKeys:
    def test_none_missing_when_both_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("TAVILY_API_KEY", "y")
        assert config.missing_keys() == {}

    def test_reports_the_single_absent_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert set(config.missing_keys()) == {"TAVILY_API_KEY"}

    def test_reports_all_keys_when_none_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert set(config.missing_keys()) == {"ANTHROPIC_API_KEY", "TAVILY_API_KEY"}

    def test_present_but_empty_value_counts_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `not os.environ.get(key)` treats an empty value (e.g. `KEY=` in a .env,
        # or an unset CI secret) as missing — distinct from a membership check.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("TAVILY_API_KEY", "y")
        assert set(config.missing_keys()) == {"ANTHROPIC_API_KEY"}


class TestBuildModel:
    def test_no_sampling_params_are_set(self) -> None:
        # Guards a real footgun: Opus 5 returns HTTP 400 if temperature/top_p/
        # top_k is sent. `ChatAnthropic` omits unset params, so all must be None.
        model = config.build_model()
        assert model.temperature is None
        assert model.top_p is None
        assert model.top_k is None

    def test_uses_configured_model_and_max_tokens(self) -> None:
        model = config.build_model()
        assert model.model == config.MODEL_NAME
        assert model.max_tokens == config.MAX_TOKENS

    def test_the_model_streams_which_is_what_makes_max_tokens_safe(self) -> None:
        # `MAX_TOKENS` above 21_333 is only safe because the request itself streams.
        # The SDK refuses a NON-streaming request whose worst-case runtime exceeds ten
        # minutes (`3600 * max_tokens / 128_000 > 600`) — but only when the client still
        # carries the SDK's default timeout, and langchain hands it `timeout=None`, so
        # that guard never fires here. A too-large non-streaming request would therefore
        # not error; it would hang the REPL forever against a client with no timeout at
        # all. `streaming=True` is what keeps us out of that regime, and it must not be
        # quietly dropped (or, worse, passed as `False`, which HARD-disables streaming
        # via `_streaming_disabled()` even under a streaming `stream_mode`).
        #
        # `_should_stream` is pure Python — no network, no key. It is the real invariant.
        model = config.build_model()
        assert model._should_stream(async_api=False) is True

    def test_streaming_does_not_smuggle_in_a_sampling_param(self) -> None:
        # The two invariants meet here: streaming must buy headroom WITHOUT reopening
        # the Opus 5 400-on-temperature footgun. Assert on the actual request payload
        # rather than trusting that `streaming=True` is inert.
        payload = config.build_model()._get_request_payload(
            [{"role": "user", "content": "hi"}]
        )
        assert not {"temperature", "top_p", "top_k"} & set(payload)

    def test_thinking_is_adaptive_and_summarized_never_disabled(self) -> None:
        # Two separate invariants, both measured against the real API, both silent when
        # broken — which is why they are asserted on the built payload rather than
        # trusted from the constructor kwargs.
        #
        # 1. `display` MUST be "summarized". Opus 5 defaults it to "omitted", and an
        #    omitted-display thinking block comes back as `{"signature", "type"}` with
        #    NO `thinking` text — so `langchain_anthropic` cannot send it back, and the
        #    replay 400s with `messages.N.content.M.thinking.thinking: Field required`.
        #    That is not a multi-turn-only bug: every tool result replays the assistant
        #    message that requested the call, so one research turn trips it as soon as
        #    `tavily_search` returns. "summarized" costs nothing extra (display controls
        #    visibility, not billing) and never reaches the user (`cli._text_of` takes
        #    bare strings and `{"type": "text"}` blocks; a thinking block is neither).
        #
        # 2. It MUST NOT be "disabled". On Opus 5 that makes the model occasionally emit
        #    a tool call as plain *text* instead of a `tool_use` block: the turn
        #    succeeds, nothing raises, and the search simply never runs — a silent no-op
        #    in an agent whose entire job is searching. Lower `output_config.effort` if
        #    the bill matters; do not disable.
        #
        # Being explicit costs portability: this shape 400s on pre-4.6 models, which
        # want the removed `budget_tokens` form. A `DEEP_RESEARCH_MODEL` override that
        # far back has to drop the argument — the right trade against a broken default.
        payload = config.build_model()._get_request_payload(
            [{"role": "user", "content": "hi"}]
        )
        assert payload.get("thinking") == {"type": "adaptive", "display": "summarized"}

    def test_max_tokens_leaves_room_for_thinking_and_the_answer(self) -> None:
        # On Opus 5 `max_tokens` caps thinking PLUS the answer, and this agent's answer
        # is a long cited report. Too tight a ceiling truncates it mid-sentence with
        # `stop_reason="max_tokens"` and no exception — the failure is silent, so the
        # floor is pinned here. Safe to raise (Opus 5 allows 128k) only because the
        # request streams; see `test_the_model_streams_which_is_what_makes_max_tokens_safe`.
        assert config.MAX_TOKENS >= 64_000
