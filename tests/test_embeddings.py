"""Tests for legiscope.embeddings module."""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest

from legiscope.embeddings import (
    EmbeddingConfig,
    _build_embedding_text,
    _generate_embeddings_mistral,
    create_and_save_embeddings,
    create_embeddings_df,
    get_embeddings,
)
from legiscope.models import CodeRef, JurisdictionRef


class TestOllamaClient:
    """Test cases for ollama client usage."""

    def test_ollama_client_mocking(self):
        """Test that ollama client can be mocked properly."""
        # Create a mock client that mimics ollama.Client
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        # Should be able to use it as ollama.Client
        result = mock_client.embeddings(model="test", prompt="test")
        assert result == {"embedding": [0.1, 0.2, 0.3]}


class TestGetEmbeddings:
    """Test cases for get_embeddings function."""

    def test_get_embeddings_basic(self):
        """Test basic embedding generation."""
        # Create mock client that mimics ollama.Client (sequential processing)
        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        texts = ["text1", "text2"]
        result = get_embeddings(mock_client, texts, "test-model", "ollama")

        assert len(result) == 2
        assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert result[1].tolist() == pytest.approx([0.4, 0.5, 0.6])

        # Verify client was called correctly (sequential calls for Ollama)
        assert mock_client.embeddings.call_count == 2
        mock_client.embeddings.assert_any_call(model="test-model", prompt="text1")
        mock_client.embeddings.assert_any_call(model="test-model", prompt="text2")

    def test_get_embeddings_empty_list(self):
        """Test error handling for empty texts list."""
        mock_client = Mock()

        with pytest.raises(ValueError, match="texts parameter cannot be empty"):
            get_embeddings(mock_client, [], "test-model", "ollama")

    def test_get_embeddings_single_text(self):
        """Test embedding generation for single text."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = get_embeddings(mock_client, ["single text"], "test-model", "ollama")

        assert len(result) == 1
        assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
        mock_client.embeddings.assert_called_once_with(
            model="test-model", prompt="single text"
        )

    def test_get_embeddings_custom_model(self):
        """Test with custom model name."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        get_embeddings(mock_client, ["text"], "custom-model", "ollama")

        mock_client.embeddings.assert_called_once_with(
            model="custom-model", prompt="text"
        )

    def test_get_embeddings_client_error(self):
        """Test error handling when client fails."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            get_embeddings(mock_client, ["text"], "test-model", "ollama")

    def test_get_embeddings_none_response(self):
        """Test handling of None response from client."""
        mock_client = Mock()
        mock_client.embeddings.return_value = None

        with pytest.raises(ValueError, match="Failed to get embedding"):
            get_embeddings(mock_client, ["text"], "test-model", "ollama")

    def test_get_embeddings_missing_embedding_key(self):
        """Test handling of response without embedding key."""
        mock_client = Mock()
        mock_client.embeddings.return_value = {"other_key": "value"}

        with pytest.raises(ValueError, match="Failed to get embedding"):
            get_embeddings(mock_client, ["text"], "test-model", "ollama")

    @patch("legiscope.embeddings.logger")
    def test_get_embeddings_progress_logging(self, mock_logger):
        """Test progress logging for large batches."""
        mock_client = Mock()
        # Create sequential responses for 15 texts (Ollama processes sequentially)
        mock_client.embeddings.side_effect = [{"embedding": [0.1]} for _ in range(15)]

        texts = [f"text{i}" for i in range(15)]
        get_embeddings(mock_client, texts, "test-model", "ollama")

        # Should log individual processing progress for Ollama
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Processed 15/15 texts" in call for call in debug_calls)

    @patch("legiscope.embeddings.logger")
    @patch(
        "legiscope.config.get",
        return_value=5,
    )
    def test_get_embeddings_progress_logging_uses_config_interval(
        self, _mock_get_config, mock_logger
    ):
        """Progress logging interval should come from config.yaml."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = [{"embedding": [0.1]} for _ in range(12)]

        texts = [f"text{i}" for i in range(12)]
        get_embeddings(mock_client, texts, "test-model", "ollama")

        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Processed 5/12 texts" in call for call in debug_calls)
        assert any("Processed 10/12 texts" in call for call in debug_calls)
        assert any("Processed 12/12 texts" in call for call in debug_calls)

    @patch("legiscope.embeddings.logger")
    @patch(
        "legiscope.config.get",
        return_value=5,
    )
    def test_get_embeddings_mistral_progress_logging_uses_config_interval(
        self, _mock_get_config, mock_logger
    ):
        """Mistral batching should also use config-driven progress logging."""
        mock_client = Mock()

        responses = []
        for _ in range(6):
            response = Mock()
            response.data = [Mock(embedding=[0.1]), Mock(embedding=[0.2])]
            responses.append(response)
        mock_client.embeddings.create.side_effect = responses

        texts = [f"text{i}" for i in range(12)]
        _generate_embeddings_mistral(mock_client, texts, "mistral-embed", batch_size=2)

        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Processed 6/12 texts" in call for call in debug_calls)
        assert any("Processed 10/12 texts" in call for call in debug_calls)
        assert any("Processed 12/12 texts" in call for call in debug_calls)

    def test_get_embeddings_mistral_provider(self):
        """Test embedding generation with Mistral provider."""
        # Create mock client that mimics mistralai.Mistral
        mock_client = Mock()

        # Create mock response object with data attribute
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["text1"]
        result = get_embeddings(mock_client, texts, "mistral-embed", "mistral")

        assert len(result) == 1
        assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])

        # Verify client was called correctly for Mistral API
        mock_client.embeddings.create.assert_called_once_with(
            model="mistral-embed", inputs=["text1"]
        )

    def test_get_embeddings_auto_detect_ollama(self):
        """Test auto-detection of ollama provider."""
        # Create mock client with ollama in name
        mock_client = Mock()
        mock_client.__class__.__name__ = "OllamaClient"
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        texts = ["text"]
        result = get_embeddings(mock_client, texts)  # No provider specified

        assert len(result) == 1
        assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_get_embeddings_auto_detect_mistral(self):
        """Test auto-detection of mistral provider."""
        # Create mock client with mistral in name
        mock_client = Mock()
        mock_client.__class__.__name__ = "Mistral"

        # Create mock response object
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["text"]
        result = get_embeddings(mock_client, texts)  # No provider specified

        assert len(result) == 1
        assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_get_embeddings_auto_detect_fails(self):
        """Test auto-detection failure with unknown client type."""
        # Create a mock without any embedding-related attributes
        mock_client = Mock(spec=[])  # Empty spec means no attributes
        mock_client.__class__.__name__ = "UnknownClient"

        with pytest.raises(ValueError, match="Unable to detect provider"):
            get_embeddings(mock_client, ["text"])  # No provider specified

    def test_get_embeddings_batching_100_texts(self):
        """Test batching behavior with exactly 100 texts (Ollama sequential)."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1 * i, 0.2 * i, 0.3 * i]} for i in range(100)
        ]

        texts = [f"text{i}" for i in range(100)]
        result = get_embeddings(mock_client, texts, "test-model", "ollama")

        assert len(result) == 100
        assert mock_client.embeddings.call_count == 100

    def test_get_embeddings_batch_error_handling(self):
        """Test error handling in batched mode."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = Exception("Batch failed")

        texts = [f"text{i}" for i in range(10)]

        with pytest.raises(Exception, match="Batch failed"):
            get_embeddings(mock_client, texts, "test-model", "ollama")


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

        # Mock embedding client with sequential responses (Ollama)
        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        config = EmbeddingConfig(model="test-model", provider="ollama")
        result = create_embeddings_df(df, mock_client, config)

        # Check structure
        assert len(result) == 2
        assert "embedding" in result.columns
        assert result.columns == ["section_heading", "segment_text", "embedding"]

        # Check embeddings
        embeddings = result["embedding"].to_list()
        assert embeddings[0] == pytest.approx([0.1, 0.2, 0.3])
        assert embeddings[1] == pytest.approx([0.4, 0.5, 0.6])

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

        config = EmbeddingConfig(
            model="test-model",
            provider="ollama",
            heading_col="custom_heading",
            text_col="custom_text",
            embedding_col="custom_embedding",
        )
        result = create_embeddings_df(df, mock_client, config)

        assert "custom_embedding" in result.columns
        assert result["custom_embedding"].to_list()[0] == pytest.approx([0.1, 0.2, 0.3])

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

        config = EmbeddingConfig(model="test-model", provider="ollama")
        create_embeddings_df(df, mock_client, config)

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

        config = EmbeddingConfig(model="test-model", provider="ollama")
        create_embeddings_df(df, mock_client, config)

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

        config = EmbeddingConfig(model="test-model", provider="ollama")
        create_embeddings_df(df, mock_client, config)

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

        config = EmbeddingConfig(model="test-model", provider="ollama")
        result = create_embeddings_df(df, mock_client, config)

        assert len(result) == 0
        assert "embedding" in result.columns
        # Should not call client
        mock_client.embeddings.assert_not_called()

    def test_create_embeddings_df_invalid_dataframe_type(self):
        """Test error handling for invalid DataFrame type."""
        invalid_df = "not a dataframe"  # type: ignore
        config = EmbeddingConfig(model="test-model", provider="ollama")
        with pytest.raises(TypeError, match="df must be a polars DataFrame"):
            create_embeddings_df(invalid_df, Mock(), config)  # type: ignore

    def test_create_embeddings_df_missing_columns(self):
        """Test error handling for missing required columns."""
        df = pl.DataFrame(
            {
                "section_heading": ["# Title"],
                # Missing segment_text
            }
        )

        mock_client = Mock()
        config = EmbeddingConfig(model="test-model", provider="ollama")

        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            create_embeddings_df(df, mock_client, config)

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
        config = EmbeddingConfig(model="test-model", provider="ollama")

        with pytest.raises(Exception, match="Embedding failed"):
            create_embeddings_df(df, mock_client, config)

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
        config = EmbeddingConfig(model="test-model", provider="ollama")

        create_embeddings_df(df, mock_client, config)

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
        config = EmbeddingConfig(model="test-model", provider="ollama")

        result = create_embeddings_df(df, mock_client, config)

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
        config = EmbeddingConfig(model="test-model", provider="ollama")

        result = create_embeddings_df(df, mock_client, config)

        # Check that embedding column is List(Float32)
        schema = result.schema
        assert schema["embedding"] == pl.List(pl.Float32)

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
        config = EmbeddingConfig(model="test-model", provider="ollama")

        result = create_embeddings_df(df, mock_client, config)

        # Should preserve all original columns plus embedding
        expected_columns = [
            "section_heading",
            "segment_text",
            "extra_column",
            "embedding",
        ]
        assert set(result.columns) == set(expected_columns)
        assert result["extra_column"][0] == "extra_value"


class TestEmbeddingConfigDefaults:
    """Test that EmbeddingConfig defaults use the new ID column names."""

    def test_id_col_default(self):
        """Test that id_col defaults to segment_id."""
        config = EmbeddingConfig()
        assert config.id_col == "segment_id"

    def test_id_col_custom(self):
        """Test that id_col can be overridden."""
        config = EmbeddingConfig(id_col="custom_id")
        assert config.id_col == "custom_id"


class TestEmbeddingIndexConfigDefaults:
    """Test that EmbeddingIndexConfig defaults use the new ID column names."""

    def test_id_col_default(self):
        """Test that id_col defaults to segment_id."""
        from legiscope.embeddings import EmbeddingIndexConfig

        df = pl.DataFrame(
            {"segment_id": [0], "segment_text": ["t"], "embedding": [[0.1]]}
        )
        config = EmbeddingIndexConfig(df=df)
        assert config.id_col == "segment_id"


class TestBuildEmbeddingText:
    """Test cases for _build_embedding_text function."""

    def test_basic_with_ancestors(self):
        """Test embedding text assembly with ancestor headings."""
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_text": ["Title I", "Chapter 1", "Section 1.1"],
                "ancestor_path": [None, "0", "0/1"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "section_ordinal": [2],
                "segment_text": ["Body text here."],
            }
        )

        result = _build_embedding_text(segments_df, sections_df)

        assert len(result) == 1
        assert result[0] == "Title I\n\nChapter 1\n\nBody text here."

    def test_no_ancestors(self):
        """Test segment with no ancestor_path (root section)."""
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["Root"],
                "ancestor_path": [None],
            }
        )
        segments_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "segment_text": ["Root body."],
            }
        )

        result = _build_embedding_text(segments_df, sections_df)

        assert len(result) == 1
        assert result[0] == "Root body."

    def test_empty_segment_text(self):
        """Test segment with empty text."""
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_text": ["Title", "Section"],
                "ancestor_path": [None, "0"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "section_ordinal": [1],
                "segment_text": [""],
            }
        )

        result = _build_embedding_text(segments_df, sections_df)

        assert len(result) == 1
        # Only ancestor heading, no empty text appended
        assert result[0] == "Title"

    def test_multiple_segments(self):
        """Test with multiple segments from different sections."""
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_text": ["Title", "Sec A", "Sec B"],
                "ancestor_path": [None, "0", "0"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "section_ordinal": [1, 2],
                "segment_text": ["Body A", "Body B"],
            }
        )

        result = _build_embedding_text(segments_df, sections_df)

        assert len(result) == 2
        assert result[0] == "Title\n\nBody A"
        assert result[1] == "Title\n\nBody B"

    def test_missing_section_ordinal(self):
        """Test segment referencing a section not in sections_df."""
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["Title"],
                "ancestor_path": [None],
            }
        )
        segments_df = pl.DataFrame(
            {
                "section_ordinal": [99],
                "segment_text": ["Orphan text"],
            }
        )

        result = _build_embedding_text(segments_df, sections_df)

        assert len(result) == 1
        assert result[0] == "Orphan text"


class TestCreateAndSaveEmbeddings:
    """Test cases for create_and_save_embeddings function."""

    def _make_code_ref(self):
        """Helper to create a CodeRef for testing."""
        return CodeRef(
            jurisdiction=JurisdictionRef(state="CA", locality="TestCity"),
            code_slug="test-code",
        )

    def test_basic_workflow(self, tmp_path):
        """Test the full create-and-save workflow."""
        code_ref = self._make_code_ref()

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_text": ["Title", "Section 1"],
                "ancestor_path": [None, "0"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0, 1],
                "section_ordinal": [1, 1],
                "section_heading": ["Section 1", "Section 1"],
                "segment_text": ["Part one.", "Part two."],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        output_path = tmp_path / "embeddings.parquet"
        config = EmbeddingConfig(model="test-model", provider="ollama")

        result = create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=mock_client,
            code_ref=code_ref,
            embedding_config=config,
            output_path=output_path,
        )

        # Verify output DataFrame
        assert len(result) == 2
        assert "segment_id" in result.columns
        assert "embedding_text" in result.columns
        assert "embedding" in result.columns
        assert "code_id" in result.columns
        assert "jurisdiction_id" in result.columns

        # Verify IDs
        assert result["segment_id"][0] == code_ref.segment_id(0)
        assert result["segment_id"][1] == code_ref.segment_id(1)
        assert result["code_id"][0] == code_ref.code_id
        assert result["jurisdiction_id"][0] == code_ref.jurisdiction_id

        # Verify embedding_text contains ancestor headings
        assert "Title" in result["embedding_text"][0]
        assert "Part one." in result["embedding_text"][0]

        # Verify file was written
        assert output_path.exists()
        loaded = pl.read_parquet(output_path)
        assert len(loaded) == 2

    def test_default_output_path(self, tmp_path, monkeypatch):
        """Test that default output path uses code_ref.full_data_dir."""
        code_ref = self._make_code_ref()

        # Create the directory so the write succeeds
        output_dir = tmp_path / "data" / "laws" / "CA" / "TestCity" / "test-code"
        output_dir.mkdir(parents=True)

        # Monkeypatch laws_dir so full_data_dir resolves to tmp_path
        import legiscope.models as models_mod

        monkeypatch.setattr(models_mod, "laws_dir", lambda: tmp_path / "data" / "laws")

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["Title"],
                "ancestor_path": [None],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [0],
                "section_heading": ["Title"],
                "segment_text": ["Body."],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2]}

        config = EmbeddingConfig(model="test-model", provider="ollama")

        create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=mock_client,
            code_ref=code_ref,
            embedding_config=config,
        )

        expected_path = output_dir / "embeddings.parquet"
        assert expected_path.exists()

    def test_output_schema(self, tmp_path):
        """Test that the output has the expected columns and types."""
        code_ref = self._make_code_ref()

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["Title"],
                "ancestor_path": [None],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [0],
                "section_heading": ["Title"],
                "segment_text": ["Body."],
            }
        )

        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2]}
        config = EmbeddingConfig(model="test-model", provider="ollama")

        result = create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=mock_client,
            code_ref=code_ref,
            embedding_config=config,
            output_path=tmp_path / "embeddings.parquet",
        )

        expected_columns = [
            "segment_id",
            "segment_ordinal",
            "section_ordinal",
            "code_id",
            "jurisdiction_id",
            "section_heading",
            "segment_text",
            "embedding_text",
            "embedding",
        ]
        assert result.columns == expected_columns
        assert result.schema["segment_id"] == pl.String
        assert result.schema["embedding_text"] == pl.String
        assert result.schema["embedding"] == pl.List(pl.Float32)


class TestChromaOperations:
    """Test cases for ChromaDB operations."""

    @patch(
        "legiscope.params.load_params",
        return_value={"embeddings": {"chroma_batch_size": 1}},
    )
    def test_add_documents_to_collection_uses_params_batch_size(
        self, _mock_load_params
    ):
        """Chroma write batching should use the params-driven batch size."""
        from legiscope.embeddings import _add_documents_to_collection

        mock_collection = MagicMock()

        _add_documents_to_collection(
            collection=mock_collection,
            ids=["s0", "s1"],
            documents=["text1", "text2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadata_list=[
                {"section_heading": "Heading 1"},
                {"section_heading": "Heading 2"},
            ],
        )

        assert mock_collection.add.call_count == 2
        first_call = mock_collection.add.call_args_list[0].kwargs
        second_call = mock_collection.add.call_args_list[1].kwargs
        assert first_call["ids"] == ["s0"]
        assert second_call["ids"] == ["s1"]

    def test_get_or_create_collection(self):
        """Test getting or creating a collection."""
        from legiscope.embeddings import (
            CollectionConfig,
            get_or_create_legal_collection,
        )

        with patch("chromadb.PersistentClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # Case 1: Collection exists
            mock_collection = MagicMock()
            mock_client.get_collection.return_value = mock_collection

            config = CollectionConfig(collection_name="test_coll")
            result = get_or_create_legal_collection(config)

            assert result == mock_collection
            mock_client.get_collection.assert_called_with(name="test_coll")

            # Case 2: Collection needs creation
            mock_client.get_collection.side_effect = Exception("Not found")
            mock_client.create_collection.return_value = mock_collection

            result = get_or_create_legal_collection(config)

            mock_client.create_collection.assert_called_with(name="test_coll")

    def test_create_embedding_index(self):
        """Test creating an embedding index from DataFrame."""
        from legiscope.embeddings import EmbeddingIndexConfig, create_embedding_index

        # Create test DataFrame
        df = pl.DataFrame(
            {
                "segment_id": ["s0", "s1"],
                "segment_text": ["text1", "text2"],
                "embedding": [[0.1, 0.2], [0.3, 0.4]],
                "section_heading": ["Heading 1", "Heading 2"],
            }
        )

        with patch(
            "legiscope.embeddings.get_or_create_legal_collection"
        ) as mock_get_coll:
            mock_collection = MagicMock()
            mock_get_coll.return_value = mock_collection

            config = EmbeddingIndexConfig(
                df=df, collection_name="test_coll", jurisdiction_id="IL-Test"
            )

            create_embedding_index(config)

            # Verify data added
            mock_collection.add.assert_called()
            call_kwargs = mock_collection.add.call_args.kwargs

            assert call_kwargs["ids"] == ["s0", "s1"]
            assert call_kwargs["documents"] == ["text1", "text2"]
            assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

            # Check metadata includes jurisdiction
            metadatas = call_kwargs["metadatas"]
            assert len(metadatas) == 2
            assert metadatas[0]["jurisdiction_id"] == "IL-Test"
            assert metadatas[0]["section_heading"] == "Heading 1"

    def test_add_jurisdiction_embeddings(self):
        """Test adding jurisdiction embeddings."""
        from legiscope.embeddings import add_jurisdiction_embeddings

        df = pl.DataFrame(
            {
                "segment_id": ["s0"],
                "segment_text": ["text"],
                "embedding": [[0.1]],
                "section_heading": ["H"],
            }
        )

        mock_collection = MagicMock()
        mock_collection.name = "test_coll"

        with patch("legiscope.embeddings.create_embedding_index") as mock_create_idx:
            add_jurisdiction_embeddings(mock_collection, df, "IL-Test")

            mock_create_idx.assert_called_once()
            config = mock_create_idx.call_args[0][0]
            assert config.jurisdiction_id == "IL-Test"
            assert config.collection_name == "test_coll"

    def test_collection_config_provider_model_naming(self):
        """Test that CollectionConfig generates provider_model suffixed names."""
        from legiscope.embeddings import CollectionConfig, EMBEDDING_PROVIDER_CONFIG

        # Default provider (ollama) should auto-resolve model
        config = CollectionConfig(provider="ollama")
        expected_model = EMBEDDING_PROVIDER_CONFIG["ollama"]["model"]
        assert config.collection_name == f"legal_code_ollama_{expected_model}"
        assert config.model == expected_model

    def test_collection_config_explicit_model(self):
        """Test that an explicitly provided model overrides auto-resolution."""
        from legiscope.embeddings import CollectionConfig

        config = CollectionConfig(provider="ollama", model="custom")
        assert config.collection_name == "legal_code_ollama_custom"
        assert config.model == "custom"

    def test_collection_config_custom_base_with_provider(self):
        """Test custom base name with provider gets _{provider}_{model} suffix."""
        from legiscope.embeddings import CollectionConfig, EMBEDDING_PROVIDER_CONFIG

        expected_model = EMBEDDING_PROVIDER_CONFIG["ollama"]["model"]
        config = CollectionConfig(collection_name="my_collection", provider="ollama")
        assert config.collection_name == f"my_collection_ollama_{expected_model}"

    def test_collection_config_no_provider(self):
        """Test that without provider, collection name is unchanged."""
        from legiscope.embeddings import CollectionConfig

        config = CollectionConfig(collection_name="legal_code_all")
        assert config.collection_name == "legal_code_all"
        assert config.model is None

    def test_create_and_persist_embeddings(self):
        """Test the unified workflow."""
        from legiscope.embeddings import (
            JurisdictionConfig,
            create_and_persist_embeddings,
        )

        df = pl.DataFrame({"text": ["content"]})
        mock_client = MagicMock()

        # Mock dependencies
        with (
            patch("legiscope.embeddings.create_embeddings_df") as mock_create_df,
            patch("legiscope.embeddings.create_embedding_index") as mock_create_idx,
        ):
            # Setup mock return for create_embeddings_df
            embeddings_df = pl.DataFrame(
                {
                    "segment_id": ["s0"],
                    "segment_text": ["content"],
                    "embedding": [[0.1]],
                }
            )
            # Add write_parquet mock to the dataframe
            embeddings_df.write_parquet = MagicMock()
            mock_create_df.return_value = embeddings_df

            mock_collection = MagicMock()
            mock_create_idx.return_value = mock_collection

            # Run workflow
            jur_config = JurisdictionConfig(jurisdiction_id="IL-Test")
            result_df, result_coll = create_and_persist_embeddings(
                df, mock_client, jurisdiction_config=jur_config
            )

            # Verify steps
            mock_create_df.assert_called_once()
            embeddings_df.write_parquet.assert_called_once()  # Persistence
            mock_create_idx.assert_called_once()  # Index creation

            assert result_df is embeddings_df
            assert result_coll is mock_collection
