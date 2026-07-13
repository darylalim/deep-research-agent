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
        # Guards a real footgun: Opus 4.8 returns HTTP 400 if temperature/top_p/
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
        # the Opus 4.8 400-on-temperature footgun. Assert on the actual request payload
        # rather than trusting that `streaming=True` is inert.
        payload = config.build_model()._get_request_payload(
            [{"role": "user", "content": "hi"}]
        )
        assert not {"temperature", "top_p", "top_k"} & set(payload)
