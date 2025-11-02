"""Tests for legiscope.utils module."""

import pytest
import tempfile
from unittest.mock import Mock, patch
from pydantic import BaseModel

from legiscope.utils import ask


class TestResponseModel(BaseModel):
    """Simple test model for testing purposes."""

    name: str
    value: int


class TestAskFunction:
    """Test cases for ask function."""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Setup temporary logging for each test."""
        # Create a temporary directory for logs during testing
        self.temp_log_dir = tempfile.mkdtemp()

        # Patch the log directory in the utils module
        with patch("legiscope.utils.os.getcwd", return_value=self.temp_log_dir):
            with patch("legiscope.utils.logger") as mock_logger:
                self.mock_logger = mock_logger
                yield

        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_log_dir, ignore_errors=True)

    def test_client_none_raises_error(self):
        """Test that None client raises ValueError."""
        with pytest.raises(
            ValueError, match="Client does not appear to be an Instructor instance"
        ):
            ask(client=None, prompt="test prompt", response_model=TestResponseModel)

        # Verify error was logged
        self.mock_logger.error.assert_called_with(
            "Client validation failed - missing chat.completions attributes"
        )

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        mock_client = Mock()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            ask(client=mock_client, prompt="", response_model=TestResponseModel)

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            ask(client=mock_client, prompt="   ", response_model=TestResponseModel)

        # Verify error was logged (should be called twice)
        self.mock_logger.error.assert_called_with(
            "Prompt validation failed - empty or whitespace only"
        )

    def test_successful_call(self):
        """Test successful LLM call with structured response."""
        # Setup mock client
        mock_client = Mock()
        mock_response = TestResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=TestResponseModel,
            model="gpt-4",
            temperature=0.5,
        )

        # Verify call was made correctly
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "user", "content": "Extract name and value from this text"}
            ],
            response_model=TestResponseModel,
            model="gpt-4",
            temperature=0.5,
            max_retries=3,  # Default parameter
        )

        # Verify result
        assert result == mock_response

        # Verify debug logs were called
        self.mock_logger.debug.assert_called()

    def test_successful_call_with_system_prompt(self):
        """Test successful LLM call with system prompt."""
        # Setup mock client
        mock_client = Mock()
        mock_response = TestResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function with system prompt
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=TestResponseModel,
            system="You are a helpful assistant.",
            model="gpt-4",
        )

        # Verify call was made correctly with system prompt
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Extract name and value from this text"},
            ],
            response_model=TestResponseModel,
            model="gpt-4",
            temperature=0.1,  # Default parameter
            max_retries=3,  # Default parameter
        )

        # Verify result
        assert result == mock_response

        # Verify system prompt was logged
        self.mock_logger.debug.assert_any_call(
            "System prompt added - length: {}", len("You are a helpful assistant.")
        )

    def test_default_parameters(self):
        """Test that default parameters are used when not specified."""
        mock_client = Mock()
        mock_response = TestResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        ask(client=mock_client, prompt="test prompt", response_model=TestResponseModel)

        # Verify default parameters were used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-5-mini"
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["max_retries"] == 3

    def test_exception_handling(self):
        """Test that exceptions are properly wrapped with context."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        with pytest.raises(Exception, match="Error in ask_llm: LLM error"):
            ask(
                client=mock_client,
                prompt="test prompt",
                response_model=TestResponseModel,
            )

        # Verify error was logged
        self.mock_logger.error.assert_called()

    def test_invalid_client_raises_error(self):
        """Test that invalid client raises ValueError."""
        mock_client = Mock()
        # Remove chat.completions attributes to make it invalid
        del mock_client.chat

        with pytest.raises(
            ValueError, match="Client does not appear to be an Instructor instance"
        ):
            ask(
                client=mock_client,
                prompt="test prompt",
                response_model=TestResponseModel,
            )
