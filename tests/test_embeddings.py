"""Tests for legiscope.embeddings module."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from legiscope.embeddings import (
    EmbeddingConfig,
    _generate_embeddings_mistral,
    _normalize_chroma_metadata,
    create_and_save_embeddings,
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
    def test_get_embeddings_no_per_segment_logging(self, mock_logger):
        """Embedding functions should not log per-segment; progress is in _embed_with_fallback."""
        mock_client = Mock()
        mock_client.embeddings.side_effect = [{"embedding": [0.1]} for _ in range(15)]

        texts = [f"text{i}" for i in range(15)]
        get_embeddings(mock_client, texts, "test-model", "ollama")

        # No info or debug calls should reference individual segment progress
        all_calls = [
            call[0][0]
            for call in mock_logger.info.call_args_list
            + mock_logger.debug.call_args_list
        ]
        assert not any("Processed" in c for c in all_calls)

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


class TestIsContextLengthError:
    """Unit tests for _is_context_length_error."""

    def test_matches_context_length(self):
        """Recognises 'context length' substring (case-insensitive)."""
        from legiscope.embeddings import _is_context_length_error

        assert _is_context_length_error(ValueError("Exceeds context length limit"))

    def test_matches_input_length(self):
        """Recognises 'input length' substring."""
        from legiscope.embeddings import _is_context_length_error

        assert _is_context_length_error(ValueError("input length exceeds the maximum"))

    def test_rejects_unrelated_error(self):
        """Unrelated errors return False."""
        from legiscope.embeddings import _is_context_length_error

        assert not _is_context_length_error(RuntimeError("connection refused"))

    def test_rejects_empty_message(self):
        """Empty exception message returns False."""
        from legiscope.embeddings import _is_context_length_error

        assert not _is_context_length_error(ValueError(""))


class TestCompactAncestorHeadings:
    """Unit tests for _compact_ancestor_headings."""

    def test_empty_input(self):
        """Empty heading list returns empty list."""
        from legiscope.embeddings import _compact_ancestor_headings

        assert _compact_ancestor_headings([], 100, reserve_body_token=False) == []

    def test_all_fit(self):
        """When all headings fit, they are returned in document order (root→leaf)."""
        from legiscope.embeddings import _compact_ancestor_headings

        headings = ["Title I", "Chapter 2", "Section 3"]
        result = _compact_ancestor_headings(headings, 100, reserve_body_token=False)
        assert result == headings

    def test_document_order_preserved(self):
        """Output order matches input (root→leaf), not reversed."""
        from legiscope.embeddings import _compact_ancestor_headings

        headings = ["Root", "Mid", "Leaf"]
        result = _compact_ancestor_headings(headings, 100, reserve_body_token=False)
        assert result[0] == "Root"
        assert result[-1] == "Leaf"

    def test_nearest_headings_preferred(self):
        """When budget is tight, nearest ancestors (leaf-side) are kept."""
        from legiscope.embeddings import _compact_ancestor_headings

        far = " ".join(["far"] * 12)  # ~12 tokens
        mid = "mid"  # ~1 token
        near = "near"  # ~1 token
        # Budget = 3 tokens — only mid + near should fit
        result = _compact_ancestor_headings(
            [far, mid, near], 3, reserve_body_token=False
        )
        assert near in result
        assert mid in result
        assert far not in result

    def test_nearest_truncated_when_alone_exceeds(self):
        """If the nearest heading alone exceeds budget, it is truncated."""
        from legiscope.embeddings import _compact_ancestor_headings

        long = " ".join(["word"] * 30)
        result = _compact_ancestor_headings([long], 5, reserve_body_token=False)
        assert len(result) == 1
        # The truncated heading should be shorter than the original
        assert len(result[0].split()) < len(long.split())

    def test_reserve_body_token_reduces_budget(self):
        """reserve_body_token=True leaves 1 token for the body."""
        from legiscope.embeddings import _compact_ancestor_headings

        # Budget = 2, reserve 1 for body → only 1 token for headings
        result_reserved = _compact_ancestor_headings(
            ["A", "B"], 2, reserve_body_token=True
        )
        result_full = _compact_ancestor_headings(
            ["A", "B"], 2, reserve_body_token=False
        )
        # With reservation, fewer heading tokens available
        assert len(result_reserved) <= len(result_full)


class TestSplitSegmentRow:
    """Unit tests for _split_segment_row."""

    @staticmethod
    def _sections_by_ordinal(sections_data):
        return {row["section_ordinal"]: row for row in sections_data}

    def test_no_split_when_under_limit(self):
        """Row within token limit returns single-element lists."""
        from legiscope.embeddings import _split_segment_row

        row = {"section_ordinal": 1, "segment_text": "Short body.", "word_count": 2}
        sections = self._sections_by_ordinal(
            [
                {"section_ordinal": 0, "heading_text": "Root", "ancestor_path": None},
                {"section_ordinal": 1, "heading_text": "Sec", "ancestor_path": "0"},
            ]
        )
        rows, texts = _split_segment_row(row, sections, 100)
        assert len(rows) == 1
        assert len(texts) == 1
        # Heading + body should be present
        assert "Root" in texts[0]
        assert "Short body." in texts[0]

    def test_splits_oversized_body(self):
        """When assembled text exceeds limit, body is split into chunks."""
        from legiscope.embeddings import _split_segment_row

        long_body = " ".join(["word"] * 200)
        row = {"section_ordinal": 1, "segment_text": long_body, "word_count": 200}
        sections = self._sections_by_ordinal(
            [
                {"section_ordinal": 0, "heading_text": "Title", "ancestor_path": None},
                {"section_ordinal": 1, "heading_text": "Child", "ancestor_path": "0"},
            ]
        )
        rows, texts = _split_segment_row(row, sections, 20)
        assert len(rows) > 1
        assert len(texts) == len(rows)
        # Each chunk should include the ancestor heading
        for t in texts:
            assert "Title" in t

    def test_halve_budget_produces_more_splits(self):
        """halve_budget=True should produce more, smaller chunks."""
        from legiscope.embeddings import _split_segment_row

        body = " ".join(["word"] * 100)
        row = {"section_ordinal": 0, "segment_text": body, "word_count": 100}
        sections = self._sections_by_ordinal(
            [{"section_ordinal": 0, "heading_text": "T", "ancestor_path": None}]
        )
        rows_normal, _ = _split_segment_row(row, sections, 20)
        rows_halved, _ = _split_segment_row(row, sections, 20, halve_budget=True)
        assert len(rows_halved) >= len(rows_normal)

    def test_heading_compaction_on_tight_budget(self):
        """When headings alone approach the limit, compaction is triggered."""
        from legiscope.embeddings import _split_segment_row

        far_heading = " ".join(["far"] * 20)
        near_heading = "near"
        row = {"section_ordinal": 2, "segment_text": "Body.", "word_count": 1}
        sections = self._sections_by_ordinal(
            [
                {
                    "section_ordinal": 0,
                    "heading_text": far_heading,
                    "ancestor_path": None,
                },
                {
                    "section_ordinal": 1,
                    "heading_text": near_heading,
                    "ancestor_path": "0",
                },
                {
                    "section_ordinal": 2,
                    "heading_text": "Leaf",
                    "ancestor_path": "0/1",
                },
            ]
        )
        rows, texts = _split_segment_row(row, sections, 10)
        # Near heading should be kept, far heading dropped
        assert near_heading in texts[0]
        assert far_heading not in texts[0]

    def test_document_order_in_assembled_text(self):
        """Headings in assembled text appear root→leaf, not reversed."""
        from legiscope.embeddings import _split_segment_row

        row = {"section_ordinal": 2, "segment_text": "Body.", "word_count": 1}
        sections = self._sections_by_ordinal(
            [
                {
                    "section_ordinal": 0,
                    "heading_text": "Root",
                    "ancestor_path": None,
                },
                {
                    "section_ordinal": 1,
                    "heading_text": "Mid",
                    "ancestor_path": "0",
                },
                {
                    "section_ordinal": 2,
                    "heading_text": "Leaf",
                    "ancestor_path": "0/1",
                },
            ]
        )
        _, texts = _split_segment_row(row, sections, 100)
        # Root should appear before Mid in the assembled text
        assert texts[0].index("Root") < texts[0].index("Mid")

    def test_no_section_ordinal(self):
        """Row without section_ordinal still works (no headings)."""
        from legiscope.embeddings import _split_segment_row

        row = {"segment_text": "Some text.", "word_count": 2}
        _, texts = _split_segment_row(row, {}, 100)
        assert texts == ["Some text."]


class TestSplitOversizedEmbeddingSegments:
    """Unit tests for _split_oversized_embedding_segments."""

    def test_no_split_needed(self):
        """When all segments fit, returns original DataFrame unchanged."""
        from legiscope.embeddings import _split_oversized_embedding_segments

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
                "segment_text": ["Short text."],
                "word_count": [2],
            }
        )
        result_df, texts = _split_oversized_embedding_segments(
            segments_df, sections_df, 100
        )
        # Original DataFrame returned as-is (identity)
        assert result_df is segments_df
        assert len(texts) == 1

    def test_splits_oversized_and_renumbers(self):
        """Oversized segment is split and segment_ordinal renumbered sequentially."""
        from legiscope.embeddings import _split_oversized_embedding_segments

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["T"],
                "ancestor_path": [None],
            }
        )
        long_body = " ".join(["word"] * 200)
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [0],
                "segment_text": [long_body],
                "word_count": [200],
            }
        )
        result_df, texts = _split_oversized_embedding_segments(
            segments_df, sections_df, 20
        )
        assert len(result_df) > 1
        assert len(texts) == len(result_df)
        # segment_ordinal should be sequential 0, 1, 2, ...
        ordinals = result_df["segment_ordinal"].to_list()
        assert ordinals == list(range(len(ordinals)))

    def test_mixed_rows_only_oversized_split(self):
        """Only the oversized segment is split; the short one passes through."""
        from legiscope.embeddings import _split_oversized_embedding_segments

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_text": ["T", "S"],
                "ancestor_path": [None, "0"],
            }
        )
        long_body = " ".join(["word"] * 200)
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0, 1],
                "section_ordinal": [0, 1],
                "segment_text": ["Short.", long_body],
                "word_count": [1, 200],
            }
        )
        result_df, texts = _split_oversized_embedding_segments(
            segments_df, sections_df, 20
        )
        # More rows than original (the long one was split)
        assert len(result_df) > 2
        # First text should be the short segment's assembled text
        assert "Short." in texts[0]


class TestEmbeddingConfigDefaults:
    """Unit tests for EmbeddingConfig default resolution."""

    def test_default_provider_resolves_to_module_constant(self):
        """EmbeddingConfig() should resolve provider to EMBEDDING_PROVIDER."""
        from legiscope.embeddings import EMBEDDING_PROVIDER

        config = EmbeddingConfig()
        assert config.provider == EMBEDDING_PROVIDER

    def test_explicit_provider_preserved(self):
        """Explicitly setting provider keeps the given value."""
        config = EmbeddingConfig(provider="mistral")
        assert config.provider == "mistral"

    def test_invalid_provider_raises(self):
        """Unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            EmbeddingConfig(provider="nonexistent")

    def test_default_model_is_none(self):
        """Default model is None (resolved later by provider config)."""
        config = EmbeddingConfig()
        assert config.model is None


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

    def test_propagates_chunk_metadata_and_region_headings(self, tmp_path):
        """Chunk metadata and fallback headings should survive embedding generation."""
        code_ref = self._make_code_ref()

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Article I"],
                "ancestor_path": [None],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [-1],
                "section_heading": ["Legal Intro"],
                "segment_text": ["This ordinance was adopted by the council."],
                "chunk_ordinal": [0],
                "chunk_id": ["CA:TestCity:test-code:c0"],
                "source_kind": ["region"],
                "region_role": ["legal_intro"],
                "context_path": ["Legal Intro"],
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

        assert "chunk_id" in result.columns
        assert "source_kind" in result.columns
        assert result["chunk_id"][0] == "CA:TestCity:test-code:c0"
        assert "Legal Intro" in result["embedding_text"][0]


class TestFallbackSplittingOnContextError:
    """Test that context-length errors on individual segments trigger splitting.

    The fallback logic in ``_embed_with_fallback`` processes texts in the
    provider's native chunk size (1 for Ollama, 100 for Mistral).  When a
    segment fails, it is split and retried without discarding other work.
    """

    def _make_code_ref(self):
        return CodeRef(
            jurisdiction=JurisdictionRef(state="CA", locality="TestCity"),
            code_slug="test-code",
        )

    def test_ollama_splits_failing_segment(self, tmp_path):
        """Ollama (chunk_size=1): failing segment is split in place."""
        code_ref = self._make_code_ref()

        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_text": ["Title", "Section 1"],
                "ancestor_path": [None, "0"],
            }
        )
        long_text = " ".join(["word"] * 300)
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0, 1],
                "section_ordinal": [1, 1],
                "section_heading": ["Section 1", "Section 1"],
                "segment_text": ["Short.", long_text],
            }
        )

        first_fail_done = False

        def mock_get_embeddings(client, texts, model, provider):
            import numpy as np

            nonlocal first_fail_done
            assert len(texts) == 1, "Ollama should process one text at a time"
            # Fail the first time we see the full long text
            if not first_fail_done and len(texts[0].split()) > 200:
                first_fail_done = True
                raise ValueError("input length exceeds the context length")
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        config = EmbeddingConfig(model="test-model", provider="ollama")

        with patch(
            "legiscope.embeddings.get_embeddings", side_effect=mock_get_embeddings
        ):
            result = create_and_save_embeddings(
                segments_df=segments_df,
                sections_df=sections_df,
                client=Mock(),
                code_ref=code_ref,
                embedding_config=config,
                output_path=tmp_path / "embeddings.parquet",
                embedding_model_token_limit=1024,
            )

        assert len(result) >= 2
        assert "embedding" in result.columns

    def test_mistral_batch_failure_falls_back(self, tmp_path):
        """Mistral (chunk_size=100): batch failure retries per-segment in chunk."""
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
                "segment_text": ["Good.", " ".join(["word"] * 300)],
            }
        )

        calls: list[int] = []  # track len(texts) per call

        def mock_get_embeddings(client, texts, model, provider):
            import numpy as np

            calls.append(len(texts))
            # Multi-text batch call fails
            if len(texts) > 1:
                raise ValueError("input length exceeds the context length")
            # Single-text calls succeed
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        config = EmbeddingConfig(model="test-model", provider="mistral")

        # Patch the provider config so chunk_size=100 is used
        patched_cfg = {
            "mistral": {
                "batch_size": 100,
                "embedding_function": None,  # not used — get_embeddings is mocked
                "model": "test-model",
                "client_factory": None,
            }
        }

        with (
            patch(
                "legiscope.embeddings.get_embeddings", side_effect=mock_get_embeddings
            ),
            patch("legiscope.embeddings.EMBEDDING_PROVIDER_CONFIG", patched_cfg),
        ):
            result = create_and_save_embeddings(
                segments_df=segments_df,
                sections_df=sections_df,
                client=Mock(),
                code_ref=code_ref,
                embedding_config=config,
                output_path=tmp_path / "embeddings.parquet",
                embedding_model_token_limit=1024,
            )

        # First call was a batch (2 texts), then individual calls
        assert calls[0] == 2
        assert all(c == 1 for c in calls[1:])
        assert len(result) >= 2
        assert "embedding" in result.columns

    def test_non_context_error_not_retried(self, tmp_path):
        """Non-context errors are raised immediately, no retry."""
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

        config = EmbeddingConfig(model="test-model", provider="ollama")

        with patch(
            "legiscope.embeddings.get_embeddings",
            side_effect=ValueError("some other error"),
        ):
            with pytest.raises(ValueError, match="some other error"):
                create_and_save_embeddings(
                    segments_df=segments_df,
                    sections_df=sections_df,
                    client=Mock(),
                    code_ref=code_ref,
                    embedding_config=config,
                    output_path=tmp_path / "embeddings.parquet",
                    embedding_model_token_limit=1024,
                )

    def test_max_retries_per_segment_exhausted(self, tmp_path):
        """After max per-segment retries, raises even if it's a context error."""
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

        config = EmbeddingConfig(model="test-model", provider="ollama")

        with patch(
            "legiscope.embeddings.get_embeddings",
            side_effect=ValueError("input length exceeds the context length"),
        ):
            with pytest.raises(ValueError, match="still exceeds context length"):
                create_and_save_embeddings(
                    segments_df=segments_df,
                    sections_df=sections_df,
                    client=Mock(),
                    code_ref=code_ref,
                    embedding_config=config,
                    output_path=tmp_path / "embeddings.parquet",
                    embedding_model_token_limit=1024,
                )

    def test_compacts_headings_when_ancestors_exceed_limit(self, tmp_path):
        """Nearest ancestor headings are preserved when context must be compacted."""
        code_ref = self._make_code_ref()

        far_heading = " ".join(["far"] * 12)
        near_heading = " ".join(["near"] * 8)
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_text": [far_heading, near_heading, "Section 1"],
                "ancestor_path": [None, "0", "0/1"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [2],
                "section_heading": ["Section 1"],
                "segment_text": [" ".join(["body"] * 6)],
            }
        )

        config = EmbeddingConfig(model="test-model", provider="ollama")
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=mock_client,
            code_ref=code_ref,
            embedding_config=config,
            output_path=tmp_path / "embeddings.parquet",
            embedding_model_token_limit=10,
        )

        assert len(result) >= 1
        embedding_texts = result["embedding_text"].to_list()
        assert any(near_heading in text for text in embedding_texts)
        assert all(far_heading not in text for text in embedding_texts)

    def test_truncates_nearest_heading_when_single_heading_exceeds_limit(
        self, tmp_path
    ):
        """If only the nearest heading fits, it is truncated to the heading budget."""
        code_ref = self._make_code_ref()

        long_heading = " ".join(["heading"] * 40)
        sections_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_text": [long_heading, "Section 1"],
                "ancestor_path": [None, "0"],
            }
        )
        segments_df = pl.DataFrame(
            {
                "segment_ordinal": [0],
                "section_ordinal": [1],
                "section_heading": ["Section 1"],
                "segment_text": ["Body."],
            }
        )

        config = EmbeddingConfig(model="test-model", provider="ollama")
        mock_client = Mock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        result = create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=mock_client,
            code_ref=code_ref,
            embedding_config=config,
            output_path=tmp_path / "embeddings.parquet",
            embedding_model_token_limit=10,
        )

        embedding_texts = result["embedding_text"].to_list()
        assert any(text.startswith("heading") for text in embedding_texts)
        assert all(long_heading not in text for text in embedding_texts)


class TestChromaOperations:
    """Test cases for ChromaDB operations."""

    def test_normalize_chroma_metadata(self):
        """Metadata should be reduced to ChromaDB-compatible scalar values."""
        metadata = {
            "null_value": None,
            "int_value": np.int64(7),
            "float_value": np.float32(1.25),
            "nan_value": float("nan"),
            "bool_value": np.bool_(True),
            "date_value": date(2026, 4, 16),
            "path_value": Path("data/laws"),
            "list_value": [1, "two"],
        }

        normalized = _normalize_chroma_metadata(metadata)

        assert normalized == {
            "int_value": 7,
            "float_value": pytest.approx(1.25),
            "bool_value": True,
            "date_value": "2026-04-16",
            "path_value": "data/laws",
            "list_value": '[1, "two"]',
        }

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

    def test_create_embedding_index_drops_null_metadata_values(self):
        """Null metadata values should be removed before Chroma insertion."""
        from legiscope.embeddings import EmbeddingIndexConfig, create_embedding_index

        df = pl.DataFrame(
            {
                "segment_id": ["s0", "s1"],
                "segment_text": ["text1", "text2"],
                "embedding": [[0.1, 0.2], [0.3, 0.4]],
                "section_heading": ["Heading 1", "Heading 2"],
                "section_number": [None, "1.0"],
                "context_path": [None, "Part 1 > Section 1"],
                "retrieval_priority": [1, 2],
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

            metadatas = mock_collection.add.call_args.kwargs["metadatas"]
            assert "section_number" not in metadatas[0]
            assert "context_path" not in metadatas[0]
            assert metadatas[0]["retrieval_priority"] == 1
            assert metadatas[1]["section_number"] == "1.0"
            assert metadatas[1]["context_path"] == "Part 1 > Section 1"

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
