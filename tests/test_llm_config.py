"""
Tests for the llm_config module — params.yaml-driven configuration.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import instructor
import pytest

from legiscope.llm_config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "llm": {
        "default_provider": "mistral",
        "providers": {
            "openai": {"fast": "gpt-4.1-mini", "powerful": "gpt-4.1"},
            "litellm": {
                "fast": "gemini/gemini-3.5-flash",
                "powerful": "gemini/gemini-3.5-flash",
            },
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


def _provider_config_from_params(params: dict) -> dict:
    """Build llm_config.PROVIDER_CONFIG-like mapping from test params."""
    providers = params.get("llm", {}).get("providers", {})
    return {
        name: {
            "fast_model": models.get("fast", ""),
            "powerful_model": models.get("powerful", ""),
            "mode": None,
            "num_ctx": models.get("num_ctx"),
        }
        for name, models in providers.items()
    }


@pytest.fixture(autouse=True)
def _patch_provider_config():
    """Keep tests hermetic from real params.yaml import-time state."""
    with patch(
        "legiscope.llm_config._get_provider_config",
        return_value=_provider_config_from_params(_BASE_PARAMS),
    ):
        yield


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
            assert (
                Config.get_fast_model()
                == _BASE_PARAMS["llm"]["providers"]["mistral"]["fast"]
            )

    def test_default_powerful_model(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            assert (
                Config.get_powerful_model()
                == _BASE_PARAMS["llm"]["providers"]["mistral"]["powerful"]
            )


class TestProviderSwitch:
    """Test switching providers via params."""

    def test_openai_provider_models(self):
        p = _params_with(**{"llm.default_provider": "openai"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_llm_provider() == "openai"
            assert (
                Config.get_fast_model()
                == _BASE_PARAMS["llm"]["providers"]["openai"]["fast"]
            )
            assert (
                Config.get_powerful_model()
                == _BASE_PARAMS["llm"]["providers"]["openai"]["powerful"]
            )

    def test_ollama_provider_models(self):
        p = _params_with(**{"llm.default_provider": "ollama"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_llm_provider() == "ollama"
            assert (
                Config.get_fast_model()
                == _BASE_PARAMS["llm"]["providers"]["ollama"]["fast"]
            )
            assert (
                Config.get_powerful_model()
                == _BASE_PARAMS["llm"]["providers"]["ollama"]["powerful"]
            )

    def test_litellm_provider_models(self):
        p = _params_with(**{"llm.default_provider": "litellm"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_llm_provider() == "litellm"
            assert (
                Config.get_fast_model()
                == _BASE_PARAMS["llm"]["providers"]["litellm"]["fast"]
            )
            assert (
                Config.get_powerful_model()
                == _BASE_PARAMS["llm"]["providers"]["litellm"]["powerful"]
            )


class TestLiteLLMClient:
    def test_litellm_gemini_client_uses_direct_provider_env(self):
        p = _params_with(**{"llm.default_provider": "litellm"})
        fake_completion = Mock()

        with (
            patch("legiscope.llm_config.load_params", return_value=p),
            patch(
                "legiscope.llm_config.get_config",
                side_effect=lambda key, default=None: {
                    "llm.litellm.api_base": None,
                    "llm.litellm.api_key_env": None,
                }.get(key, default),
            ),
            patch.dict(
                sys.modules,
                {"litellm": SimpleNamespace(completion=fake_completion)},
            ),
            patch("legiscope.llm_config.instructor.from_litellm") as mock_from_litellm,
        ):
            Config.get_fast_client()

        completion_partial = mock_from_litellm.call_args.args[0]
        assert completion_partial.func is fake_completion
        assert completion_partial.keywords == {"model": "gemini/gemini-3.5-flash"}
        assert mock_from_litellm.call_args.kwargs["mode"] == instructor.Mode.TOOLS

    def test_litellm_client_uses_partial_completion_defaults(self):
        p = _params_with(**{"llm.default_provider": "litellm"})
        fake_completion = Mock()

        with (
            patch("legiscope.llm_config.load_params", return_value=p),
            patch(
                "legiscope.llm_config.get_config",
                side_effect=lambda key, default=None: {
                    "llm.litellm.api_base": "http://localhost:4000",
                    "llm.litellm.api_key_env": "LITELLM_GATEWAY_KEY",
                }.get(key, default),
            ),
            patch.dict(
                sys.modules,
                {"litellm": SimpleNamespace(completion=fake_completion)},
            ),
            patch.dict("os.environ", {"LITELLM_GATEWAY_KEY": "secret"}),
            patch("legiscope.llm_config.instructor.from_litellm") as mock_from_litellm,
        ):
            Config.get_fast_client()

        completion_partial = mock_from_litellm.call_args.args[0]
        assert completion_partial.func is fake_completion
        assert completion_partial.keywords["model"] == "gemini/gemini-3.5-flash"
        assert completion_partial.keywords["api_base"] == "http://localhost:4000"
        assert completion_partial.keywords["api_key"] == "secret"
        assert mock_from_litellm.call_args.kwargs["mode"] == instructor.Mode.TOOLS


class TestOpenAIServedModel:
    def test_openai_served_model_uses_matching_params_value(self):
        p = _params_with(
            **{
                "llm.default_provider": "openai",
                "llm.providers.openai.fast": "gpt-4.1",
            }
        )
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_openai_served_model() == "gpt-4.1"

    def test_openai_served_model_allows_single_defined_model(self):
        p = _params_with(
            **{
                "llm.default_provider": "openai",
                "llm.providers.openai.fast": "",
            }
        )
        with patch("legiscope.llm_config.load_params", return_value=p):
            assert Config.get_openai_served_model() == "gpt-4.1"

    def test_openai_served_model_rejects_mismatched_fast_and_powerful(self):
        p = _params_with(**{"llm.default_provider": "openai"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            with pytest.raises(ValueError, match="serves only one model per job"):
                Config.get_openai_served_model()

    def test_openai_served_model_requires_openai_provider(self):
        with patch("legiscope.llm_config.load_params", return_value=_BASE_PARAMS):
            with pytest.raises(ValueError, match="llm.default_provider"):
                Config.get_openai_served_model()


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
        patched_config = _provider_config_from_params(p)
        patched_config["ollama"] = dict(patched_config.get("ollama", {}))
        patched_config["ollama"]["num_ctx"] = 8192

        with (
            patch("legiscope.llm_config.load_params", return_value=p),
            patch(
                "legiscope.llm_config._get_provider_config", return_value=patched_config
            ),
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

    def test_litellm_uses_num_retries(self):
        p = _params_with(**{"llm.default_provider": "litellm"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            params = Config.get_llm_params(model="gpt-5.5-2026-04-23")
            assert params["num_retries"] == 3
            assert "max_retries" not in params
            assert "temperature" not in params

    def test_litellm_keeps_nondefault_temperature(self):
        p = _params_with(**{"llm.default_provider": "litellm", "llm.temperature": 0.7})
        with patch("legiscope.llm_config.load_params", return_value=p):
            params = Config.get_llm_params()
            assert params["temperature"] == 0.7
            assert params["num_retries"] == 3

    def test_litellm_omits_explicit_zero_temperature_override_for_gpt5(self):
        p = _params_with(**{"llm.default_provider": "litellm"})
        with patch("legiscope.llm_config.load_params", return_value=p):
            params = Config.get_llm_params(model="gpt-5.5-2026-04-23", temperature=0.0)
            assert "temperature" not in params
            assert params["num_retries"] == 3
