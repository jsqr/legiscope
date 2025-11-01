"""Tests for legiscope.convert module."""

from unittest.mock import Mock
from pydantic import BaseModel

from legiscope.convert import (
    ask,
    BooleanResult,
)


class TestResponseModel(BaseModel):
    """Simple test model for testing purposes."""

    name: str
    value: int


class TestConvertModule:
    """Test cases for convert module functionality."""

    def test_ask_function_import(self):
        """Test that ask function is properly imported from utils."""
        # Test that we can import ask from convert module
        from legiscope.utils import ask as utils_ask
        from legiscope.convert import ask as convert_ask

        # Both should be same function
        assert utils_ask is convert_ask

    def test_ask_function_backward_compatibility(self):
        """Test that ask function works as expected when imported from convert."""
        # Setup mock client
        mock_client = Mock()
        mock_response = TestResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function imported from convert module
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


class TestResponseModels:
    """Test cases for the predefined response models."""

    def test_boolean_result_model(self):
        """Test BooleanResult model validation."""
        result = BooleanResult(
            answer=True, explanation="This is clearly true based on the evidence."
        )
        assert result.answer is True
        assert result.explanation == "This is clearly true based on the evidence."

        # Test with None answer
        result_none = BooleanResult(
            answer=None,
            explanation="The evidence is insufficient to determine a clear answer.",
        )
        assert result_none.answer is None
        assert (
            result_none.explanation
            == "The evidence is insufficient to determine a clear answer."
        )
