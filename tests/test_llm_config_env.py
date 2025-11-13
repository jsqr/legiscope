"""
Tests for the llm_config module environment variable functionality.
"""

import os
from legiscope.llm_config import Config


class TestEnvironmentVariables:
    """Test environment variable configuration for LLM models."""

    def setup_method(self):
        """Clean up environment variables before each test."""
        # Remove any existing legiscope environment variables
        env_vars = [
            "LEGISCOPE_LLM_PROVIDER",
            "LEGISCOPE_FAST_MODEL",
            "LEGISCOPE_POWERFUL_MODEL",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Clean up environment variables after each test."""
        self.setup_method()  # Reuse cleanup logic

    def test_default_provider_and_models(self):
        """Test default behavior without environment variables."""
        assert Config.LLM_PROVIDER == "mistral"
        assert Config.get_fast_model() == "mistral-medium-latest"
        assert Config.get_powerful_model() == "magistral-medium-latest"

    def test_provider_override_only(self):
        """Test provider override without model overrides."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"

        assert Config.LLM_PROVIDER == "openai"
        assert Config.get_fast_model() == "gpt-4.1-mini"
        assert Config.get_powerful_model() == "gpt-4.1"

    def test_fast_model_override_only(self):
        """Test fast model override without provider change."""
        os.environ["LEGISCOPE_FAST_MODEL"] = "custom-fast-model"

        assert Config.LLM_PROVIDER == "mistral"  # Default provider
        assert Config.get_fast_model() == "custom-fast-model"
        assert (
            Config.get_powerful_model() == "magistral-medium-latest"
        )  # Default powerful

    def test_powerful_model_override_only(self):
        """Test powerful model override without provider change."""
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = "custom-powerful-model"

        assert Config.LLM_PROVIDER == "mistral"  # Default provider
        assert Config.get_fast_model() == "mistral-medium-latest"  # Default fast
        assert Config.get_powerful_model() == "custom-powerful-model"

    def test_both_model_overrides(self):
        """Test both fast and powerful model overrides."""
        os.environ["LEGISCOPE_FAST_MODEL"] = "custom-fast"
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = "custom-powerful"

        assert Config.LLM_PROVIDER == "mistral"  # Default provider
        assert Config.get_fast_model() == "custom-fast"
        assert Config.get_powerful_model() == "custom-powerful"

    def test_provider_and_model_overrides(self):
        """Test provider override with model overrides."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"
        os.environ["LEGISCOPE_FAST_MODEL"] = "override-fast"
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = "override-powerful"

        assert Config.LLM_PROVIDER == "openai"
        assert Config.get_fast_model() == "override-fast"
        assert Config.get_powerful_model() == "override-powerful"

    def test_provider_override_with_fast_model_only(self):
        """Test provider override with only fast model override."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"
        os.environ["LEGISCOPE_FAST_MODEL"] = "override-fast"

        assert Config.LLM_PROVIDER == "openai"
        assert Config.get_fast_model() == "override-fast"
        assert Config.get_powerful_model() == "gpt-4.1"  # Default OpenAI powerful

    def test_provider_override_with_powerful_model_only(self):
        """Test provider override with only powerful model override."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = "override-powerful"

        assert Config.LLM_PROVIDER == "openai"
        assert Config.get_fast_model() == "gpt-4.1-mini"  # Default OpenAI fast
        assert Config.get_powerful_model() == "override-powerful"

    def test_mistral_provider_with_overrides(self):
        """Test Mistral provider with model overrides."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "mistral"
        os.environ["LEGISCOPE_FAST_MODEL"] = "custom-mistral-fast"
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = "custom-mistral-powerful"

        assert Config.LLM_PROVIDER == "mistral"
        assert Config.get_fast_model() == "custom-mistral-fast"
        assert Config.get_powerful_model() == "custom-mistral-powerful"

    def test_empty_string_environment_variables(self):
        """Test that empty string environment variables are not used."""
        os.environ["LEGISCOPE_FAST_MODEL"] = ""
        os.environ["LEGISCOPE_POWERFUL_MODEL"] = ""

        # Empty strings should be treated as "not set" and fall back to defaults
        assert Config.get_fast_model() == "mistral-medium-latest"
        assert Config.get_powerful_model() == "magistral-medium-latest"

    def test_environment_variable_priority(self):
        """Test that environment variables take priority over provider defaults."""
        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"
        os.environ["LEGISCOPE_FAST_MODEL"] = "should-win"

        # Environment variable should override OpenAI default
        assert Config.get_fast_model() == "should-win"
        assert Config.get_powerful_model() == "gpt-4.1"  # Still uses OpenAI default
