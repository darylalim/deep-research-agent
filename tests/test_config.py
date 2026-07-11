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
