"""Tests for legiscope.pipeline.segment — params loading and delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl

from legiscope.pipeline import segment as pipeline_segment


class TestSegmentMain:
    def test_calls_with_default_params(self, mock_cli_args, sample_code_ref):
        """Uses default token_limit=256 and words_per_token=0.75 when params are empty."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))

        with (
            patch(
                "legiscope.pipeline.segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.segment.load_params", return_value={}),
            patch("legiscope.pipeline.segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        mock_segment.assert_called_once()
        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 256
        assert kwargs["words_per_token"] == 0.75

    def test_uses_custom_token_limit(self, mock_cli_args, sample_code_ref):
        """Custom token_limit from params is forwarded."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))
        params = {"segmentation": {"token_limit": 128}}

        with (
            patch(
                "legiscope.pipeline.segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.segment.load_params", return_value=params),
            patch("legiscope.pipeline.segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 128
        assert kwargs["words_per_token"] == 0.75

    def test_uses_custom_words_per_token(self, mock_cli_args, sample_code_ref):
        """Custom words_per_token from params is forwarded."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))
        params = {"segmentation": {"words_per_token": 0.5}}

        with (
            patch(
                "legiscope.pipeline.segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.segment.load_params", return_value=params),
            patch("legiscope.pipeline.segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        _, kwargs = mock_segment.call_args
        assert kwargs["words_per_token"] == 0.5

    def test_per_code_params_override(self, mock_cli_args, sample_code_ref):
        """Per-code params.yaml override works for both segmentation keys."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))
        params = {"segmentation": {"token_limit": 64, "words_per_token": 0.9}}

        with (
            patch(
                "legiscope.pipeline.segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.segment.load_params", return_value=params),
            patch("legiscope.pipeline.segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        _, kwargs = mock_segment.call_args
        assert kwargs["token_limit"] == 64
        assert kwargs["words_per_token"] == 0.9

    def test_passes_code_ref_to_segment(self, mock_cli_args, sample_code_ref):
        """The CodeRef is passed as the first positional arg."""
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
        mock_segment = MagicMock(return_value=(pl.DataFrame(), pl.DataFrame()))

        with (
            patch(
                "legiscope.pipeline.segment.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("legiscope.pipeline.segment.load_params", return_value={}),
            patch("legiscope.pipeline.segment.segment_legal_code", mock_segment),
        ):
            pipeline_segment.main()

        args, _ = mock_segment.call_args
        assert args[0] is sample_code_ref
