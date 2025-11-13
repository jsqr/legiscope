"""Tests for legiscope.utils module."""

import pytest
from unittest.mock import Mock
from pydantic import BaseModel

from legiscope.utils import ask


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
            temperature=0.1,  # Default parameter
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
            temperature=0.1,
            max_retries=3,
        )
