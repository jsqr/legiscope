"""Tests for legiscope.pipeline.init — jurisdiction/code registration."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from legiscope.models import (
    JURISDICTIONS_SCHEMA,
    CodeRef,
    JurisdictionRef,
)
from legiscope.pipeline import init as pipeline_init


# ---------------------------------------------------------------------------
# _load_or_create_parquet
# ---------------------------------------------------------------------------


class TestLoadOrCreateParquet:
    def test_creates_empty_df_when_missing(self, tmp_path):
        path = tmp_path / "missing.parquet"
        df = pipeline_init._load_or_create_parquet(path, JURISDICTIONS_SCHEMA)
        assert df.height == 0
        assert set(df.columns) == set(JURISDICTIONS_SCHEMA.keys())

    def test_loads_existing_parquet(self, tmp_path):
        path = tmp_path / "existing.parquet"
        original = pl.DataFrame(
            [
                {
                    "jurisdiction_id": "CA",
                    "state": "CA",
                    "locality": None,
                    "level": "state",
                    "name": "California",
                    "parent_jurisdiction": None,
                }
            ],
            schema=JURISDICTIONS_SCHEMA,
        )
        original.write_parquet(path)

        df = pipeline_init._load_or_create_parquet(path, JURISDICTIONS_SCHEMA)
        assert df.height == 1
        assert df["jurisdiction_id"][0] == "CA"


# ---------------------------------------------------------------------------
# _append_jurisdiction
# ---------------------------------------------------------------------------


class TestAppendJurisdiction:
    def test_appends_new_jurisdiction(self, tmp_path):
        parquet = tmp_path / "jurisdictions.parquet"
        ref = JurisdictionRef(state="IL", locality="WindyTown")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "City of WindyTown")

        df = pl.read_parquet(parquet)
        assert df.height == 1
        assert df["jurisdiction_id"][0] == "IL-WindyTown"
        assert df["parent_jurisdiction"][0] == "IL"

    def test_skips_duplicate(self, tmp_path):
        parquet = tmp_path / "jurisdictions.parquet"
        ref = JurisdictionRef(state="IL", locality="WindyTown")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "City of WindyTown")
            pipeline_init._append_jurisdiction(ref, "City of WindyTown")

        df = pl.read_parquet(parquet)
        assert df.height == 1

    def test_state_level_no_parent(self, tmp_path):
        parquet = tmp_path / "jurisdictions.parquet"
        ref = JurisdictionRef(state="CA")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "California")

        df = pl.read_parquet(parquet)
        assert df["parent_jurisdiction"][0] is None
        assert df["level"][0] == "state"

    def test_local_sets_parent_to_state(self, tmp_path):
        parquet = tmp_path / "jurisdictions.parquet"
        ref = JurisdictionRef(state="CA", locality="LosAngeles")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "City of Los Angeles")

        df = pl.read_parquet(parquet)
        assert df["parent_jurisdiction"][0] == "CA"


# ---------------------------------------------------------------------------
# _append_code
# ---------------------------------------------------------------------------


class TestAppendCode:
    def test_appends_new_code(self, tmp_path):
        parquet = tmp_path / "codes.parquet"
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="IL", locality="WindyTown"),
            code_slug="municipal-code",
        )

        with patch.object(pipeline_init, "CODES_PARQUET", parquet):
            pipeline_init._append_code(
                code_ref, "WindyTown Municipal Code", "municipal"
            )

        df = pl.read_parquet(parquet)
        assert df.height == 1
        assert df["code_id"][0] == "IL:WindyTown:municipal-code"
        assert df["code_type"][0] == "municipal"

    def test_skips_duplicate(self, tmp_path):
        parquet = tmp_path / "codes.parquet"
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="IL", locality="WindyTown"),
            code_slug="municipal-code",
        )

        with patch.object(pipeline_init, "CODES_PARQUET", parquet):
            pipeline_init._append_code(
                code_ref, "WindyTown Municipal Code", "municipal"
            )
            pipeline_init._append_code(
                code_ref, "WindyTown Municipal Code", "municipal"
            )

        df = pl.read_parquet(parquet)
        assert df.height == 1

    def test_correct_fields(self, tmp_path):
        parquet = tmp_path / "codes.parquet"
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="CA", locality="LosAngeles"),
            code_slug="zoning-code",
        )

        with patch.object(pipeline_init, "CODES_PARQUET", parquet):
            pipeline_init._append_code(code_ref, "LA Zoning Code", "zoning")

        df = pl.read_parquet(parquet)
        assert df["jurisdiction_id"][0] == "CA-LosAngeles"
        assert df["code_slug"][0] == "zoning-code"
        assert df["level"][0] == "local"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_creates_directory_and_registries(self, tmp_path, mock_cli_args):
        mock_cli_args([])
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"
        code_dir = tmp_path / "laws" / "IL" / "WindyTown" / "municipal-code"

        fake_params = {
            "jurisdiction": {
                "state": "IL",
                "locality": "WindyTown",
                "code_slug": "municipal-code",
                "code_name": "WindyTown Municipal Code",
            }
        }

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=code_dir
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        c_df = pl.read_parquet(c_parquet)
        assert j_df.height == 1
        assert c_df.height == 1

    def test_auto_generates_jurisdiction_name(self, tmp_path, mock_cli_args):
        mock_cli_args([])
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"

        fake_params = {
            "jurisdiction": {
                "state": "IL",
                "locality": "WindyTown",
                "code_slug": "municipal-code",
                "code_name": "WindyTown Municipal Code",
            }
        }

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        assert j_df["name"][0] == "City of WindyTown"

    def test_state_level_jurisdiction_name(self, tmp_path, mock_cli_args):
        mock_cli_args([])
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"

        fake_params = {
            "jurisdiction": {
                "state": "CA",
                "locality": "State",
                "code_slug": "penal-code",
                "code_name": "California Penal Code",
            }
        }

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        assert j_df["name"][0] == "CA"
        assert j_df["level"][0] == "state"

    def test_defaults_from_params_yaml(self, tmp_path, mock_cli_args):
        """main() reads jurisdiction defaults from params.yaml (no CLI flags)."""
        mock_cli_args([])
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"

        fake_params = {
            "jurisdiction": {
                "state": "TX",
                "locality": "Austin",
                "code_slug": "city-code",
                "code_name": "Austin City Code",
            }
        }

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        c_df = pl.read_parquet(c_parquet)
        assert j_df["jurisdiction_id"][0] == "TX-Austin"
        assert c_df["name"][0] == "Austin City Code"
        assert c_df["code_slug"][0] == "city-code"

    def test_code_type_flag(self, tmp_path, mock_cli_args):
        """--code-type CLI flag is respected."""
        mock_cli_args(["--code-type", "zoning"])
        c_parquet = tmp_path / "codes.parquet"

        fake_params = {
            "jurisdiction": {
                "state": "CA",
                "locality": "LosAngeles",
                "code_slug": "zoning-code",
                "code_name": "LA Zoning Code",
            }
        }

        with (
            patch.object(
                pipeline_init, "JURISDICTIONS_PARQUET", tmp_path / "j.parquet"
            ),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        c_df = pl.read_parquet(c_parquet)
        assert c_df["code_type"][0] == "zoning"

    def test_jurisdiction_name_flag(self, tmp_path, mock_cli_args):
        """--jurisdiction-name CLI flag overrides auto-generation."""
        mock_cli_args(["--jurisdiction-name", "My Custom Name"])
        j_parquet = tmp_path / "jurisdictions.parquet"

        fake_params = {
            "jurisdiction": {
                "state": "CA",
                "locality": "LosAngeles",
                "code_slug": "mc",
                "code_name": "LA MC",
            }
        }

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", tmp_path / "c.parquet"),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        assert j_df["name"][0] == "My Custom Name"

    def test_errors_when_code_name_missing(self, mock_cli_args):
        """main() errors when code_name is not set in params.yaml."""
        mock_cli_args([])

        fake_params = {
            "jurisdiction": {
                "state": "CA",
                "code_slug": "mc",
            }
        }

        with (
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
            pytest.raises(SystemExit),
        ):
            pipeline_init.main()

    def test_errors_when_state_missing(self, mock_cli_args):
        """main() errors when state is not set in params.yaml."""
        mock_cli_args([])

        fake_params = {
            "jurisdiction": {
                "code_slug": "mc",
                "code_name": "Test Code",
            }
        }

        with (
            patch("legiscope.pipeline.init.load_params", return_value=fake_params),
            pytest.raises((SystemExit, ValueError)),
        ):
            pipeline_init.main()
