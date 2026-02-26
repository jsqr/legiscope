"""Tests for scripts/parse.py — thin wrapper delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import parse as pipeline_parse


class TestParseMain:
    def test_calls_convert_to_markdown(self, mock_cli_args, sample_code_ref):
        """Delegates to convert_to_markdown with the constructed CodeRef."""
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
        mock_convert = MagicMock()

        with (
            patch(
                "parse.CodeRef.from_dvc_vars",
                return_value=sample_code_ref,
            ),
            patch("parse.convert_to_markdown", mock_convert),
        ):
            pipeline_parse.main()

        mock_convert.assert_called_once_with(sample_code_ref)

    def test_locality_defaults_to_none(self, mock_cli_args):
        """When --locality is omitted, CodeRef.from_dvc_vars gets locality=None."""
        mock_cli_args(
            [
                "--state",
                "CA",
                "--code-slug",
                "penal-code",
            ]
        )
        mock_convert = MagicMock()
        mock_from_dvc = MagicMock()

        with (
            patch("parse.CodeRef.from_dvc_vars", mock_from_dvc),
            patch("parse.convert_to_markdown", mock_convert),
        ):
            pipeline_parse.main()

        _, kwargs = mock_from_dvc.call_args
        assert kwargs.get("locality") is None

    def test_passes_correct_code_ref_fields(self, mock_cli_args):
        """State, locality, and code_slug are forwarded to CodeRef.from_dvc_vars."""
        mock_cli_args(
            [
                "--state",
                "NY",
                "--locality",
                "NewYork",
                "--code-slug",
                "admin-code",
            ]
        )
        mock_convert = MagicMock()
        mock_from_dvc = MagicMock()

        with (
            patch("parse.CodeRef.from_dvc_vars", mock_from_dvc),
            patch("parse.convert_to_markdown", mock_convert),
        ):
            pipeline_parse.main()

        _, kwargs = mock_from_dvc.call_args
        assert kwargs["state"] == "NY"
        assert kwargs["locality"] == "NewYork"
        assert kwargs["code_slug"] == "admin-code"
