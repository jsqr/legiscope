"""Tests for scripts/index.py replacement-style ChromaDB indexing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import index as pipeline_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings_df(segment_ids: list[str]) -> pl.DataFrame:
    """Create a minimal embeddings DataFrame with the given segment IDs."""
    n = len(segment_ids)
    return pl.DataFrame(
        {
            "segment_id": segment_ids,
            "code_id": ["IL:WindyTown:municipal-code"] * n,
            "jurisdiction_id": ["IL-WindyTown"] * n,
            "embedding": [[0.1, 0.2, 0.3]] * n,
        }
    )


def _mock_collection(existing_ids: list[str]) -> MagicMock:
    """Create a mock ChromaDB collection with configurable existing IDs."""
    collection = MagicMock()
    collection.get.return_value = {"ids": existing_ids}
    collection.count.return_value = len(existing_ids)
    return collection


# ---------------------------------------------------------------------------
# Replacement indexing logic
# ---------------------------------------------------------------------------


class TestReplacementIndexing:
    def test_replaces_existing_code_segments(
        self, tmp_path, mock_cli_args, sample_code_ref
    ):
        """Existing rows for a code are deleted before re-adding current embeddings."""
        all_ids = ["IL:WindyTown:municipal-code:g0", "IL:WindyTown:municipal-code:g1"]
        embeddings_df = _make_embeddings_df(all_ids)
        embeddings_path = tmp_path / "embeddings.parquet"
        embeddings_df.write_parquet(embeddings_path)

        collection = _mock_collection(all_ids)
        mock_add = MagicMock()

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyTown",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "index.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
            patch(
                "index.chroma_db_path",
                return_value=tmp_path / "chroma_db",
            ),
            patch(
                "index.get_or_create_legal_collection",
                return_value=collection,
            ),
            patch("index.create_embedding_index", mock_add),
            patch("index.CodeRef.from_dvc_vars") as mock_from_dvc,
        ):
            mock_from_dvc.return_value = sample_code_ref
            # Point full_data_dir at tmp_path
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_index.main()

        collection.delete.assert_called_once_with(
            where={"code_id": sample_code_ref.code_id}
        )
        mock_add.assert_called_once()
        config = mock_add.call_args[0][0]
        assert config.df["segment_id"].to_list() == all_ids

    def test_reindexes_all_segments_even_when_ids_overlap(
        self, tmp_path, mock_cli_args, sample_code_ref
    ):
        """Replacement mode re-adds all rows for a code regardless of existing IDs."""
        all_ids = [
            "IL:WindyTown:municipal-code:g0",
            "IL:WindyTown:municipal-code:g1",
            "IL:WindyTown:municipal-code:g2",
        ]
        existing_ids = ["IL:WindyTown:municipal-code:g0"]
        embeddings_df = _make_embeddings_df(all_ids)
        embeddings_path = tmp_path / "embeddings.parquet"
        embeddings_df.write_parquet(embeddings_path)

        collection = _mock_collection(existing_ids)
        mock_add = MagicMock()

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyTown",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "index.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
            patch(
                "index.chroma_db_path",
                return_value=tmp_path / "chroma_db",
            ),
            patch(
                "index.get_or_create_legal_collection",
                return_value=collection,
            ),
            patch("index.create_embedding_index", mock_add),
            patch("index.CodeRef.from_dvc_vars") as mock_from_dvc,
        ):
            mock_from_dvc.return_value = sample_code_ref
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_index.main()

        collection.delete.assert_called_once_with(
            where={"code_id": sample_code_ref.code_id}
        )
        mock_add.assert_called_once()
        config = mock_add.call_args[0][0]
        added_ids = set(config.df["segment_id"].to_list())
        assert added_ids == set(all_ids)

    def test_empty_collection_adds_all(self, tmp_path, mock_cli_args, sample_code_ref):
        """Empty collection still performs replacement and adds all segments."""
        all_ids = ["IL:WindyTown:municipal-code:g0", "IL:WindyTown:municipal-code:g1"]
        embeddings_df = _make_embeddings_df(all_ids)
        embeddings_path = tmp_path / "embeddings.parquet"
        embeddings_df.write_parquet(embeddings_path)

        collection = _mock_collection([])
        mock_add = MagicMock()

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyTown",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "index.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
            patch(
                "index.chroma_db_path",
                return_value=tmp_path / "chroma_db",
            ),
            patch(
                "index.get_or_create_legal_collection",
                return_value=collection,
            ),
            patch("index.create_embedding_index", mock_add),
            patch("index.CodeRef.from_dvc_vars") as mock_from_dvc,
        ):
            mock_from_dvc.return_value = sample_code_ref
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_index.main()

        collection.delete.assert_called_once_with(
            where={"code_id": sample_code_ref.code_id}
        )
        mock_add.assert_called_once()
        config = mock_add.call_args[0][0]
        assert len(config.df) == 2

    def test_empty_embeddings_delete_existing_without_reinserting(
        self, tmp_path, mock_cli_args, sample_code_ref
    ):
        """Replacement mode removes stale rows even when no new embeddings exist."""
        embeddings_df = _make_embeddings_df([])
        embeddings_path = tmp_path / "embeddings.parquet"
        embeddings_df.write_parquet(embeddings_path)

        collection = _mock_collection(["IL:WindyTown:municipal-code:g0"])
        mock_add = MagicMock()

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyTown",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "index.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
            patch(
                "index.chroma_db_path",
                return_value=tmp_path / "chroma_db",
            ),
            patch(
                "index.get_or_create_legal_collection",
                return_value=collection,
            ),
            patch("index.create_embedding_index", mock_add),
            patch("index.CodeRef.from_dvc_vars") as mock_from_dvc,
        ):
            mock_from_dvc.return_value = sample_code_ref
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_index.main()

        collection.delete.assert_called_once_with(
            where={"code_id": sample_code_ref.code_id}
        )
        mock_add.assert_not_called()

    def test_missing_embeddings_raises(self, tmp_path, mock_cli_args, sample_code_ref):
        """FileNotFoundError when embeddings.parquet is missing."""
        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyTown",
                "--code-slug",
                "municipal-code",
            ]
        )

        collection = _mock_collection([])

        with (
            patch(
                "index.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
            patch(
                "index.chroma_db_path",
                return_value=tmp_path / "chroma_db",
            ),
            patch(
                "index.get_or_create_legal_collection",
                return_value=collection,
            ),
            patch("index.CodeRef.from_dvc_vars") as mock_from_dvc,
        ):
            mock_from_dvc.return_value = sample_code_ref
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            with pytest.raises(FileNotFoundError, match="Embeddings not found"):
                pipeline_index.main()

        collection.delete.assert_not_called()
