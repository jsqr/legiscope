"""Tests for legiscope.pipeline.embed — file checks and delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from legiscope.pipeline import embed as pipeline_embed


class TestEmbedMain:
    def test_raises_on_missing_sections(self, tmp_path, mock_cli_args, sample_code_ref):
        """FileNotFoundError when sections.parquet is missing."""
        # Write only segments (no sections)
        (tmp_path / "segments.parquet").touch()

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "legiscope.pipeline.embed.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch(
                "legiscope.pipeline.embed.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
        ):
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            with pytest.raises(FileNotFoundError, match="sections.parquet"):
                pipeline_embed.main()

    def test_raises_on_missing_segments(self, tmp_path, mock_cli_args, sample_code_ref):
        """FileNotFoundError when segments.parquet is missing."""
        # Write only sections (no segments)
        sections = pl.DataFrame(
            {
                "section_id": ["s0"],
                "code_id": ["c0"],
                "title": ["t"],
                "level": [1],
                "ordinal": [0],
                "parent_section_id": [None],
            }
        )
        sections.write_parquet(tmp_path / "sections.parquet")

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
            ]
        )

        with (
            patch(
                "legiscope.pipeline.embed.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch(
                "legiscope.pipeline.embed.load_params",
                return_value={"embeddings": {"default_provider": "mistral"}},
            ),
        ):
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            with pytest.raises(FileNotFoundError, match="segments.parquet"):
                pipeline_embed.main()

    def test_calls_create_and_save_embeddings(
        self,
        tmp_path,
        mock_cli_args,
        sample_code_ref,
        sample_sections_df,
        sample_segments_df,
    ):
        """Delegates to create_and_save_embeddings with correct args."""
        sample_sections_df.write_parquet(tmp_path / "sections.parquet")
        sample_segments_df.write_parquet(tmp_path / "segments.parquet")

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=pl.DataFrame())

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
            ]
        )

        params = {"embeddings": {"default_provider": "ollama"}}

        with (
            patch(
                "legiscope.pipeline.embed.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.embed.load_params", return_value=params),
            patch(
                "legiscope.pipeline.embed.get_embedding_client",
                return_value=mock_client,
            ),
            patch(
                "legiscope.pipeline.embed.get_default_model",
                return_value="embeddinggemma",
            ),
            patch("legiscope.pipeline.embed.create_and_save_embeddings", mock_create),
        ):
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_embed.main()

        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["client"] is mock_client
        assert kwargs["code_ref"] is sample_code_ref
        assert len(kwargs["segments_df"]) == len(sample_segments_df)
        assert len(kwargs["sections_df"]) == len(sample_sections_df)

    def test_uses_provider_from_params(
        self,
        tmp_path,
        mock_cli_args,
        sample_code_ref,
        sample_sections_df,
        sample_segments_df,
    ):
        """Provider is read from params and passed to get_embedding_client."""
        sample_sections_df.write_parquet(tmp_path / "sections.parquet")
        sample_segments_df.write_parquet(tmp_path / "segments.parquet")

        mock_get_client = MagicMock(return_value=MagicMock())
        mock_get_model = MagicMock(return_value="embeddinggemma")

        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
            ]
        )

        params = {"embeddings": {"default_provider": "ollama"}}

        with (
            patch(
                "legiscope.pipeline.embed.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.embed.load_params", return_value=params),
            patch("legiscope.pipeline.embed.get_embedding_client", mock_get_client),
            patch("legiscope.pipeline.embed.get_default_model", mock_get_model),
            patch(
                "legiscope.pipeline.embed.create_and_save_embeddings",
                return_value=pl.DataFrame(),
            ),
        ):
            type(sample_code_ref).full_data_dir = property(lambda self: tmp_path)

            pipeline_embed.main()

        mock_get_client.assert_called_once_with("ollama")
        mock_get_model.assert_called_once_with("ollama")
