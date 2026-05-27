"""Tests for legiscope.utils module."""

import argparse
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from legiscope.models import CodeRef, JurisdictionRef
from legiscope.utils import (
    LLMConfig,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    ask,
    create_structured_completion,
    create_code_structure,
    create_jurisdiction_structure,
    resolve_model_default,
    str2bool,
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

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_successful_call(self, mock_get_llm_params):
        """Test successful LLM call with structured response."""
        mock_get_llm_params.return_value = {
            "temperature": 0.5,
            "max_retries": 3,
            "model": "gpt-4",
        }
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

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_successful_call_with_system_prompt(self, mock_get_llm_params):
        """Test successful LLM call with system prompt."""
        mock_get_llm_params.return_value = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_retries": DEFAULT_MAX_RETRIES,
            "model": "gpt-4",
        }
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
            temperature=DEFAULT_TEMPERATURE,  # Default parameter
            max_retries=DEFAULT_MAX_RETRIES,  # Default parameter
        )

        # Verify result
        assert result == mock_response

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_exception_handling(self, mock_get_llm_params):
        """Test that exceptions are properly passed through."""
        mock_get_llm_params.return_value = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        with pytest.raises(Exception, match="LLM error"):
            ask(
                client=mock_client,
                prompt="test prompt",
                response_model=MockResponseModel,
            )

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_default_parameters(self, mock_get_llm_params):
        """Test that default parameters are applied correctly."""
        mock_get_llm_params.return_value = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function without specifying model/temperature
        ask(
            client=mock_client,
            prompt="test prompt",
            response_model=MockResponseModel,
        )

        # Verify defaults were applied (no model parameter since client handles it)
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "test prompt"}],
            response_model=MockResponseModel,
            temperature=DEFAULT_TEMPERATURE,
            max_retries=DEFAULT_MAX_RETRIES,
        )

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_none_kwargs_are_not_forwarded(self, mock_get_llm_params):
        mock_get_llm_params.return_value = {
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        ask(
            client=mock_client,
            prompt="test prompt",
            response_model=MockResponseModel,
            model="gpt-5.5-2026-04-23",
            temperature=None,
        )

        mock_get_llm_params.assert_called_once_with(model="gpt-5.5-2026-04-23")

    @patch("legiscope.utils.time.sleep")
    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_retries_on_429_error(self, mock_get_llm_params, mock_sleep):
        """ask() should back off and retry on 429-style provider errors."""
        mock_get_llm_params.return_value = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_retries": 2,
        }
        mock_client = Mock()
        mock_response = MockResponseModel(name="retried", value=7)
        mock_client.chat.completions.create.side_effect = [
            Exception("429 Too Many Requests"),
            mock_response,
        ]

        result = ask(
            client=mock_client,
            prompt="test prompt",
            response_model=MockResponseModel,
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(2.0)

    @patch("legiscope.utils.time.sleep")
    def test_create_structured_completion_uses_retry_after_header(self, mock_sleep):
        """Provider retry-after headers should override exponential backoff."""
        mock_client = Mock()
        mock_response = MockResponseModel(name="retried", value=9)

        retry_exc = Exception("Too Many Requests")
        retry_exc.response = Mock(headers={"retry-after": "4.5"})
        mock_client.chat.completions.create.side_effect = [retry_exc, mock_response]

        result = create_structured_completion(
            client=mock_client,
            messages=[{"role": "user", "content": "hi"}],
            response_model=MockResponseModel,
            max_retries=1,
        )

        assert result == mock_response
        mock_sleep.assert_called_once_with(4.5)

    def test_create_structured_completion_retries_without_temperature_when_rejected(self):
        """Explicit temperature should be retried once with provider defaults."""
        mock_client = Mock()
        mock_response = MockResponseModel(name="fallback", value=11)
        mock_client.chat.completions.create.side_effect = [
            Exception(
                "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported."
            ),
            mock_response,
        ]

        result = create_structured_completion(
            client=mock_client,
            messages=[{"role": "user", "content": "hi"}],
            response_model=MockResponseModel,
            temperature=0.0,
            max_retries=1,
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert first_call.kwargs["temperature"] == 0.0
        assert "temperature" not in second_call.kwargs

    def test_create_structured_completion_retries_when_temperature_is_deprecated(self):
        """Anthropic models may reject explicit temperature as deprecated."""
        mock_client = Mock()
        mock_response = MockResponseModel(name="fallback", value=12)
        mock_client.chat.completions.create.side_effect = [
            Exception("AnthropicException - {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"`temperature` is deprecated for this model.\"}}"),
            mock_response,
        ]

        result = create_structured_completion(
            client=mock_client,
            messages=[{"role": "user", "content": "hi"}],
            response_model=MockResponseModel,
            temperature=0.0,
            max_retries=1,
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert first_call.kwargs["temperature"] == 0.0
        assert "temperature" not in second_call.kwargs

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_ask_retries_without_temperature_when_provider_rejects_it(
        self, mock_get_llm_params
    ):
        mock_get_llm_params.return_value = {
            "temperature": 0.0,
            "max_retries": 1,
            "model": "gpt-5.5",
        }
        mock_client = Mock()
        mock_response = MockResponseModel(name="fallback", value=13)
        mock_client.chat.completions.create.side_effect = [
            Exception(
                "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported."
            ),
            mock_response,
        ]

        result = ask(
            client=mock_client,
            prompt="test prompt",
            response_model=MockResponseModel,
            model="gpt-5.5",
            temperature=0.0,
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert first_call.kwargs["temperature"] == 0.0
        assert "temperature" not in second_call.kwargs


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
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_retries == DEFAULT_MAX_RETRIES

    @patch("legiscope.llm_config.Config.get_fast_model")
    @patch("legiscope.params.load_params", side_effect=FileNotFoundError)
    def test_minimal_config_uses_safe_fallbacks_without_params(
        self, _mock_load_params, mock_get_fast_model
    ):
        """Missing params.yaml should not break default config creation."""
        mock_get_fast_model.return_value = "fallback-model"
        mock_client = Mock()

        config = LLMConfig(client=mock_client)

        assert config.model == "fallback-model"
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_retries == DEFAULT_MAX_RETRIES

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

    def test_valid_creation(self, tmp_path, monkeypatch):
        """Test successful structure creation."""
        data_root = tmp_path / "data"
        monkeypatch.setenv("LEGISCOPE_DATA_DIR", str(data_root))

        path = create_jurisdiction_structure("IL", "WindyTown")

        laws_path = data_root / "laws" / "IL" / "WindyTown"
        output_path = data_root / "output" / "IL-WindyTown"

        assert path == str(laws_path)
        assert laws_path.is_dir()
        assert output_path.is_dir()

    def test_empty_inputs(self):
        """Test validation of empty inputs."""
        with pytest.raises(ValueError, match="State cannot be empty"):
            create_jurisdiction_structure("", "City")

        with pytest.raises(ValueError, match="Locality cannot be empty"):
            create_jurisdiction_structure("State", "")

    def test_invalid_characters(self):
        """Test validation of alphanumeric characters."""
        with pytest.raises(ValueError, match="State must contain only alphanumeric"):
            create_jurisdiction_structure("CA!", "LosAngeles")

        with pytest.raises(ValueError, match="Locality must contain only alphanumeric"):
            create_jurisdiction_structure(
                "CA", "Los Angeles!"
            )  # spaces not allowed in validation logic?
            # actually logic says: locality.strip().replace(" ", "") then check isalnum
            # So "Los Angeles" is valid. "Los Angeles!" is invalid.

    def test_os_error_handling(self):
        """Test handling of OS errors."""
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Failed to create directory structure"):
                create_jurisdiction_structure("IL", "WindyTown")


class TestCreateCodeStructure:
    """Test create_code_structure function."""

    def test_creates_code_and_output_directories(self, tmp_path, monkeypatch):
        data_root = tmp_path / "data"
        monkeypatch.setenv("LEGISCOPE_DATA_DIR", str(data_root))
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="IL", locality="WindyTown"),
            code_slug="municipal-code",
        )

        code_dir = create_code_structure(code_ref)

        assert code_dir == data_root / "laws" / "IL" / "WindyTown" / "municipal-code"
        assert (code_dir / "raw").is_dir()
        assert (data_root / "output" / "IL-WindyTown").is_dir()


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
