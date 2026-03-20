"""Tests for scripts/segment.py — params loading and delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

import segment as pipeline_segment


class TestSegmentMain:
    def test_calls_with_default_params(self, mock_cli_args, sample_code_ref):
        """Uses default token_limit=1024 when params are empty."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))

        with (
            patch(
                "segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("segment.load_params", return_value={}),
            patch("segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        mock_segment.assert_called_once()
        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 1024

    def test_uses_custom_token_limit(self, mock_cli_args, sample_code_ref):
        """Custom token_limit from params is forwarded."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))
        params = {"segmentation": {"token_limit": 128}}

        with (
            patch(
                "segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("segment.load_params", return_value=params),
            patch("segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 128

    def test_per_code_params_override(self, mock_cli_args, sample_code_ref):
        """Per-code params.yaml override works for token_limit."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))
        params = {"segmentation": {"token_limit": 64}}

        with (
            patch(
                "segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("segment.load_params", return_value=params),
            patch("segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 64

    def test_passes_code_ref_to_segment(self, mock_cli_args, sample_code_ref):
        """The CodeRef is passed as the first positional arg."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))

        with (
            patch(
                "segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("segment.load_params", return_value={}),
            patch("segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        args, _ = mock_segment.call_args
        assert args[0] is sample_code_ref
