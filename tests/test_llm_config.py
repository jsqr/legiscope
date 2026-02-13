"""
Tests for the llm_config module — params.yaml-driven configuration.
"""

from unittest.mock import patch

import pytest

from legiscope.llm_config import Config, PROVIDER_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "llm": {
        "default_provider": "mistral",
        "providers": {
            "openai": {"fast": "gpt-4.1-mini", "powerful": "gpt-4.1"},
            "mistral": {"fast": "mistral-small-2506", "powerful": "mistral-large-2512"},
            "ollama": {"fast": "qwen3:8b", "powerful": "qwen3:30b", "num_ctx": None},
        },
        "temperature": 0.0,
        "max_retries": 3,
        "timeout": 300,
    },
    "embeddings": {"default_provider": "ollama"},
}


def _params_with(**overrides):
    """Return a copy of base params with nested overrides applied."""
    import copy

    p = copy.deepcopy(_BASE_PARAMS)
    for dotpath, value in overrides.items():
        keys = dotpath.split(".")
        d = p
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultProviderAndModels:
    """Test default behaviour reading from params.yaml."""

    def test_default_provider(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            assert Config.get_llm_provider() == "mistral"

    def test_default_fast_model(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            assert Config.get_fast_model() == PROVIDER_CONFIG["mistral"]["fast_model"]

    def test_default_powerful_model(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            assert (
                Config.get_powerful_model()
                == PROVIDER_CONFIG["mistral"]["powerful_model"]
            )


class TestProviderSwitch:
    """Test switching providers via params."""

    def test_openai_provider_models(self):
        p = _params_with(**{"llm.default_provider": "openai"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_llm_provider() == "openai"
            assert Config.get_fast_model() == PROVIDER_CONFIG["openai"]["fast_model"]
            assert (
                Config.get_powerful_model()
                == PROVIDER_CONFIG["openai"]["powerful_model"]
            )

    def test_ollama_provider_models(self):
        p = _params_with(**{"llm.default_provider": "ollama"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_llm_provider() == "ollama"
            assert Config.get_fast_model() == PROVIDER_CONFIG["ollama"]["fast_model"]
            assert (
                Config.get_powerful_model()
                == PROVIDER_CONFIG["ollama"]["powerful_model"]
            )


class TestUnsupportedProvider:
    def test_unsupported_raises(self):
        p = _params_with(**{"llm.default_provider": "unsupported"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                Config.get_fast_model()
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                Config.get_powerful_model()


class TestGetLLMParams:
    def test_defaults(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            params = Config.get_llm_params()
            assert params["temperature"] == 0.0
            assert params["max_retries"] == 3
            assert "extra_body" not in params

    def test_overrides(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            params = Config.get_llm_params(temperature=0.5, max_retries=5, foo="bar")
            assert params["temperature"] == 0.5
            assert params["max_retries"] == 5
            assert params["foo"] == "bar"

    def test_ollama_num_ctx(self):
        """Ollama num_ctx from params.yaml is forwarded as extra_body."""
        p = _params_with(**{"llm.default_provider": "ollama"})
        # Patch PROVIDER_CONFIG to include num_ctx
        patched_config = dict(PROVIDER_CONFIG)
        patched_config["ollama"] = dict(patched_config.get("ollama", {}))
        patched_config["ollama"]["num_ctx"] = 8192

        with (
            patch("legiscope.llm_config.load_params", return_value=p),
            patch("legiscope.llm_config.PROVIDER_CONFIG", patched_config),
        ):
            params = Config.get_llm_params()
            assert "extra_body" in params
            assert params["extra_body"]["num_ctx"] == 8192

    def test_ollama_no_num_ctx(self):
        """Ollama without num_ctx does not add extra_body."""
        p = _params_with(**{"llm.default_provider": "ollama"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            params = Config.get_llm_params()
            assert "extra_body" not in params
