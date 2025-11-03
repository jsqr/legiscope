"""Tests for legiscope.embeddings module."""

from unittest.mock import Mock, patch
import pytest
import polars as pl

from legiscope.embeddings import (
    EmbeddingClient,
    get_embeddings,
    create_embeddings_df,
)


class TestEmbeddingClient:
    """Test cases for EmbeddingClient protocol."""

    def test_embedding_client_protocol(self):
        """Test that EmbeddingClient is a proper protocol."""
        # Create a mock client that follows the protocol
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        # Should be able to use it as EmbeddingClient
        def use_client(client: EmbeddingClient):
            return client.embeddings(model="test", prompt="test")

        result = use_client(mock_client)
        assert result == {"embedding": [0.1, 0.2, 0.3]}


class TestGetEmbeddings:
    """Test cases for get_embeddings function."""

    def test_get_embeddings_basic(self):
        """Test basic embedding generation."""
        # Create mock client
        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        texts = ["text1", "text2"]
        result = get_embeddings(mock_client, texts, "test-model")

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

        # Verify client was called correctly
        assert mock_client.embeddings.call_count == 2
        mock_client.embeddings.assert_any_call(model="test-model", prompt="text1")
        mock_client.embeddings.assert_any_call(model="test-model", prompt="text2")

    def test_get_embeddings_empty_list(self):
        """Test error handling for empty texts list."""
        mock_client = Mock()

        with pytest.raises(ValueError, match="texts parameter cannot be empty"):
            get_embeddings(mock_client, [], "test-model")

    def test_get_embeddings_single_text(self):
        """Test embedding generation for single text."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = get_embeddings(mock_client, ["single text"], "test-model")

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.assert_called_once_with(
            model="test-model", prompt="single text"
        )

    def test_get_embeddings_custom_model(self):
        """Test with custom model name."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        get_embeddings(mock_client, ["text"], "custom-model")

        mock_client.embeddings.assert_called_once_with(
            model="custom-model", prompt="text"
        )

    def test_get_embeddings_client_error(self):
        """Test error handling when client fails."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            get_embeddings(mock_client, ["text"], "test-model")

    def test_get_embeddings_none_response(self):
        """Test handling of None response from client."""
        mock_client = Mock()
        mock_client.embeddings.return_value = None

        with pytest.raises(ValueError, match="Failed to get embedding"):
            get_embeddings(mock_client, ["text"], "test-model")

    def test_get_embeddings_missing_embedding_key(self):
        """Test handling of response without embedding key."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"other_key": "value"}

        with pytest.raises(KeyError):
            get_embeddings(mock_client, ["text"], "test-model")

    @patch("legiscope.embeddings.logger")
    def test_get_embeddings_progress_logging(self, mock_logger):
        """Test progress logging for large batches."""
        mock_client = Mock()
        # Create 15 responses to trigger progress logging
        mock_client.embeddings.side_effect = [{"embedding": [0.1]} for _ in range(15)]

        texts = [f"text{i}" for i in range(15)]
        get_embeddings(mock_client, texts, "test-model")

        # Should log progress at 10
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Processed 10/15 embeddings" in call for call in debug_calls)


class TestCreateEmbeddingsDf:
    """Test cases for create_embeddings_df function."""

    def test_create_embeddings_df_basic(self):
        """Test basic embeddings DataFrame creation."""
        # Create test DataFrame
        df = pl.DataFrame(
            {
                "section_heading": ["# Title 1", "# Title 2"],
                "segment_text": ["Content 1", "Content 2"],
            }
        )

        # Mock embedding client
        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        result = create_embeddings_df(df, mock_client, "test-model")

        # Check structure
        assert len(result) == 2
        assert "embedding" in result.columns
        assert result.columns == ["section_heading", "segment_text", "embedding"]

        # Check embeddings
        embeddings = result["embedding"].to_list()
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_create_embeddings_df_custom_columns(self):
        """Test with custom column names."""
        df = pl.DataFrame(
            {
                "custom_heading": ["# Title"],
                "custom_text": ["Content"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = create_embeddings_df(
            df,
            mock_client,
            heading_col="custom_heading",
            text_col="custom_text",
            embedding_col="custom_embedding",
        )

        assert "custom_embedding" in result.columns
        assert result["custom_embedding"].to_list()[0] == [0.1, 0.2, 0.3]

    def test_create_embeddings_df_concatenation(self):
        """Test text concatenation logic."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                "segment_text": ["Content"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        create_embeddings_df(df, mock_client, "test-model")

        # Should call with concatenated text
        expected_prompt = "# Title\n\nContent"
        mock_client.embeddings.assert_called_once_with(
            model="test-model", prompt=expected_prompt
        )

    def test_create_embeddings_df_heading_only(self):
        """Test with heading but no text."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title Only"],
                "segment_text": [None],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        create_embeddings_df(df, mock_client, "test-model")

        # Should call with heading only
        mock_client.embeddings.assert_called_once_with(
            model="test-model", prompt="# Title Only"
        )

    def test_create_embeddings_df_text_only(self):
        """Test with text but no heading."""
        df = pl.DataFrame(
            {
                "section_heading": [None],
                "segment_text": ["Text only"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        create_embeddings_df(df, mock_client, "test-model")

        # Should call with text only
        mock_client.embeddings.assert_called_once_with(
            model="test-model", prompt="Text only"
        )

    def test_create_embeddings_df_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pl.DataFrame(
            {
                "section_heading": [],
                "segment_text": [],
            }
        )

        mock_client = Mock()

        result = create_embeddings_df(df, mock_client, "test-model")

        assert len(result) == 0
        assert "embedding" in result.columns
        # Should not call client
        mock_client.embeddings.assert_not_called()

    def test_create_embeddings_df_invalid_dataframe_type(self):
        """Test error handling for invalid DataFrame type."""
        invalid_df = "not a dataframe"  # type: ignore
        with pytest.raises(TypeError, match="df must be a polars DataFrame"):
            create_embeddings_df(invalid_df, Mock(), "test-model")  # type: ignore

    def test_create_embeddings_df_missing_columns(self):
        """Test error handling for missing required columns."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                # Missing segment_text
            }
        )

        mock_client = Mock()

        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            create_embeddings_df(df, mock_client, "test-model")

    def test_create_embeddings_df_embedding_error(self):
        """Test error handling when embedding generation fails."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                "segment_text": ["Content"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.side_effect = Exception("Embedding failed")

        with pytest.raises(Exception, match="Embedding failed"):
            create_embeddings_df(df, mock_client, "test-model")

    @patch("legiscope.embeddings.logger")
    def test_create_embeddings_df_logging(self, mock_logger):
        """Test logging functionality."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                "segment_text": ["Content"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        create_embeddings_df(df, mock_client, "test-model")

        # Should log info messages
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any(
            "Creating embeddings DataFrame with model: test-model" in call
            for call in info_calls
        )
        assert any(
            "Successfully created embeddings DataFrame with 1 rows" in call
            for call in info_calls
        )

    def test_create_embeddings_df_large_dataset(self):
        """Test handling of larger dataset."""
        # Create DataFrame with multiple rows
        df = pl.DataFrame(
            {
                "section_heading": [f"# Title {i}" for i in range(5)],
                "segment_text": [f"Content {i}" for i in range(5)],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1 * i, 0.2 * i, 0.3 * i]} for i in range(5)
        ]

        result = create_embeddings_df(df, mock_client, "test-model")

        assert len(result) == 5
        assert len(result["embedding"].to_list()) == 5
        assert mock_client.embeddings.call_count == 5

    def test_create_embeddings_df_embedding_dtype(self):
        """Test that embedding column has correct dtype."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                "segment_text": ["Content"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = create_embeddings_df(df, mock_client, "test-model")

        # Check that embedding column is List(Float64)
        schema = result.schema
        assert schema["embedding"] == pl.List(pl.Float64)

    def test_create_embeddings_df_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                "segment_text": ["Content"],
                "extra_column": ["extra_value"],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = create_embeddings_df(df, mock_client, "test-model")

        # Should preserve all original columns plus embedding
        expected_columns = [
            "section_heading",
            "segment_text",
            "extra_column",
            "embedding",
        ]
        assert set(result.columns) == set(expected_columns)
        assert result["extra_column"][0] == "extra_value"
