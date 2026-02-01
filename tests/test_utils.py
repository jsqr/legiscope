"""Tests for legiscope.utils module."""

import pytest
import argparse
import os
from unittest.mock import Mock, patch
from pydantic import BaseModel

from legiscope.utils import (
    ask,
    str2bool,
    LLMConfig,
    create_jurisdiction_structure,
    resolve_model_default,
)


class MockResponseModel(BaseModel):
    """Simple test model for testing purposes."""

    name: str
    value: int


class TestAskFunction:
    """Test cases for ask function."""

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        mock_client = Mock()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            ask(client=mock_client, prompt="", response_model=MockResponseModel)

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            ask(client=mock_client, prompt="   ", response_model=MockResponseModel)

    def test_successful_call(self):
        """Test successful LLM call with structured response."""
        # Setup mock client
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=MockResponseModel,
            model="gpt-4",
            temperature=0.5,
        )

        # Verify call was made correctly
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "user", "content": "Extract name and value from this text"}
            ],
            response_model=MockResponseModel,
            model="gpt-4",
            temperature=0.5,
            max_retries=3,  # Default parameter
        )

        # Verify result
        assert result == mock_response

    def test_successful_call_with_system_prompt(self):
        """Test successful LLM call with system prompt."""
        # Setup mock client
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function with system prompt
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=MockResponseModel,
            system="You are a helpful assistant.",
            model="gpt-4",
        )

        # Verify call was made correctly with system prompt
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Extract name and value from this text"},
            ],
            response_model=MockResponseModel,
            model="gpt-4",
            temperature=0.0,  # Default parameter
            max_retries=3,  # Default parameter
        )

        # Verify result
        assert result == mock_response

    def test_exception_handling(self):
        """Test that exceptions are properly passed through."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        with pytest.raises(Exception, match="LLM error"):
            ask(
                client=mock_client,
                prompt="test prompt",
                response_model=MockResponseModel,
            )

    def test_default_parameters(self):
        """Test that default parameters are applied correctly."""
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function without specifying model/temperature
        ask(
            client=mock_client,
            prompt="test prompt",
            response_model=MockResponseModel,
        )

        # Verify defaults were applied
        # Verify defaults were applied (no model parameter since client handles it)
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "test prompt"}],
            response_model=MockResponseModel,
            temperature=0.0,
            max_retries=3,
        )


class TestStr2Bool:
    """Test str2bool function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("t", True),
            ("T", True),
            ("yes", True),
            ("YES", True),
            ("y", True),
            ("Y", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("f", False),
            ("F", False),
            ("no", False),
            ("NO", False),
            ("n", False),
            ("N", False),
            ("0", False),
        ],
    )
    def test_valid_inputs(self, value, expected):
        """Test valid boolean string representations."""
        assert str2bool(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "maybe",
            "2",
            "foo",
            "",
            None,
        ],
    )
    def test_invalid_inputs(self, value):
        """Test invalid inputs raise ArgumentTypeError (if user provided) or TypeError."""
        # argparse would catch this in real usage, but str2bool raises ArgumentTypeError
        # or AttributeError if not string/bool.
        if value is None:
            # This depends on implementation if it expects strict string
            # The implementation uses v.lower() so it will raise AttributeError for None
            with pytest.raises(AttributeError):
                str2bool(value)
        else:
            with pytest.raises(argparse.ArgumentTypeError):
                str2bool(value)


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with just required parameters."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client)

        assert config.client is mock_client
        assert config.model is not None  # Should be set by __post_init__
        assert config.temperature == 0.0  # Default
        assert config.max_retries == 3  # Default

    def test_explicit_model(self):
        """Test config with explicit model specified."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, model="gpt-4")

        assert config.model == "gpt-4"

    def test_custom_temperature(self):
        """Test config with custom temperature."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, temperature=0.5)

        assert config.temperature == 0.5

    def test_custom_max_retries(self):
        """Test config with custom max_retries."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, max_retries=5)

        assert config.max_retries == 5

    def test_all_custom_params(self):
        """Test config with all parameters customized."""
        mock_client = Mock()
        config = LLMConfig(
            client=mock_client, model="gpt-4-turbo", temperature=0.7, max_retries=10
        )

        assert config.client is mock_client
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.7
        assert config.max_retries == 10

    def test_temperature_validation_too_low(self):
        """Test that temperature below 0 raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(client=mock_client, temperature=-0.1)

    def test_temperature_validation_too_high(self):
        """Test that temperature above 2.0 raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(client=mock_client, temperature=2.1)

    def test_temperature_at_boundaries(self):
        """Test that boundary values for temperature are accepted."""
        mock_client = Mock()

        # Lower boundary
        config_low = LLMConfig(client=mock_client, temperature=0.0)
        assert config_low.temperature == 0.0

        # Upper boundary
        config_high = LLMConfig(client=mock_client, temperature=2.0)
        assert config_high.temperature == 2.0

    def test_max_retries_validation(self):
        """Test that negative max_retries raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LLMConfig(client=mock_client, max_retries=-1)

    def test_max_retries_zero(self):
        """Test that zero max_retries is allowed."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, max_retries=0)
        assert config.max_retries == 0

    def test_config_is_dataclass(self):
        """Test that LLMConfig behaves as a dataclass."""
        mock_client = Mock()
        config1 = LLMConfig(client=mock_client, model="gpt-4")
        config2 = LLMConfig(client=mock_client, model="gpt-4")

        # Dataclasses support equality comparison
        assert config1.model == config2.model
        assert config1.temperature == config2.temperature

    def test_config_repr(self):
        """Test that config has useful repr."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, model="test-model")
        repr_str = repr(config)

        assert "LLMConfig" in repr_str
        assert "test-model" in repr_str


class TestCreateJurisdictionStructure:
    """Test create_jurisdiction_structure function."""

    def test_valid_creation(self):
        """Test successful structure creation."""
        with patch("os.makedirs") as mock_makedirs:
            path = create_jurisdiction_structure("IL", "WindyCity")

            # Check return value
            assert path == os.path.join("data", "laws", "IL-WindyCity")

            # Check directories created
            # Base directory + 3 subdirectories = 4 calls
            assert mock_makedirs.call_count == 4

            # Check calls
            base_path = os.path.join("data", "laws", "IL-WindyCity")
            mock_makedirs.assert_any_call(base_path, exist_ok=True)
            mock_makedirs.assert_any_call(os.path.join(base_path, "raw"), exist_ok=True)
            mock_makedirs.assert_any_call(
                os.path.join(base_path, "processed"), exist_ok=True
            )
            mock_makedirs.assert_any_call(
                os.path.join(base_path, "tables"), exist_ok=True
            )

    def test_empty_inputs(self):
        """Test validation of empty inputs."""
        with pytest.raises(ValueError, match="State cannot be empty"):
            create_jurisdiction_structure("", "City")

        with pytest.raises(ValueError, match="Municipality cannot be empty"):
            create_jurisdiction_structure("State", "")

    def test_invalid_characters(self):
        """Test validation of alphanumeric characters."""
        with pytest.raises(ValueError, match="State must contain only alphanumeric"):
            create_jurisdiction_structure("CA!", "LosAngeles")

        with pytest.raises(
            ValueError, match="Municipality must contain only alphanumeric"
        ):
            create_jurisdiction_structure(
                "CA", "Los Angeles!"
            )  # spaces not allowed in validation logic?
            # actually logic says: municipality.strip().replace(" ", "") then check isalnum
            # So "Los Angeles" is valid. "Los Angeles!" is invalid.

    def test_os_error_handling(self):
        """Test handling of OS errors."""
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Failed to create directory structure"):
                create_jurisdiction_structure("IL", "WindyCity")


class TestResolveModelDefault:
    """Test resolve_model_default function."""

    def test_explicit_model(self):
        """Test when model is explicitly provided."""
        assert resolve_model_default("gpt-4", use_fast=True) == "gpt-4"
        assert resolve_model_default("gpt-4", use_fast=False) == "gpt-4"

    @patch("legiscope.llm_config.Config.get_fast_model")
    def test_default_fast_model(self, mock_get_fast):
        """Test resolving to default fast model."""
        mock_get_fast.return_value = "fast-model"

        assert resolve_model_default(None, use_fast=True) == "fast-model"
        mock_get_fast.assert_called_once()

    @patch("legiscope.llm_config.Config.get_powerful_model")
    def test_default_powerful_model(self, mock_get_powerful):
        """Test resolving to default powerful model."""
        mock_get_powerful.return_value = "powerful-model"

        assert resolve_model_default(None, use_fast=False) == "powerful-model"
        mock_get_powerful.assert_called_once()
