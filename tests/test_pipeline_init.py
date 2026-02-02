"""Tests for legiscope.pipeline.init — jurisdiction/code registration."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl

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
        ref = JurisdictionRef(state="IL", locality="WindyCity")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "City of WindyCity")

        df = pl.read_parquet(parquet)
        assert df.height == 1
        assert df["jurisdiction_id"][0] == "IL-WindyCity"
        assert df["parent_jurisdiction"][0] == "IL"

    def test_skips_duplicate(self, tmp_path):
        parquet = tmp_path / "jurisdictions.parquet"
        ref = JurisdictionRef(state="IL", locality="WindyCity")

        with patch.object(pipeline_init, "JURISDICTIONS_PARQUET", parquet):
            pipeline_init._append_jurisdiction(ref, "City of WindyCity")
            pipeline_init._append_jurisdiction(ref, "City of WindyCity")

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
            jurisdiction=JurisdictionRef(state="IL", locality="WindyCity"),
            code_slug="municipal-code",
        )

        with patch.object(pipeline_init, "CODES_PARQUET", parquet):
            pipeline_init._append_code(
                code_ref, "WindyCity Municipal Code", "municipal"
            )

        df = pl.read_parquet(parquet)
        assert df.height == 1
        assert df["code_id"][0] == "IL:WindyCity:municipal-code"
        assert df["code_type"][0] == "municipal"

    def test_skips_duplicate(self, tmp_path):
        parquet = tmp_path / "codes.parquet"
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="IL", locality="WindyCity"),
            code_slug="municipal-code",
        )

        with patch.object(pipeline_init, "CODES_PARQUET", parquet):
            pipeline_init._append_code(
                code_ref, "WindyCity Municipal Code", "municipal"
            )
            pipeline_init._append_code(
                code_ref, "WindyCity Municipal Code", "municipal"
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
        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
                "--name",
                "WindyCity Municipal Code",
            ]
        )
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"
        code_dir = tmp_path / "laws" / "IL" / "WindyCity" / "municipal-code"

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=code_dir
            ),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        c_df = pl.read_parquet(c_parquet)
        assert j_df.height == 1
        assert c_df.height == 1

    def test_auto_generates_jurisdiction_name(self, tmp_path, mock_cli_args):
        mock_cli_args(
            [
                "--state",
                "IL",
                "--locality",
                "WindyCity",
                "--code-slug",
                "municipal-code",
                "--name",
                "WindyCity Municipal Code",
            ]
        )
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        assert j_df["name"][0] == "City of WindyCity"

    def test_state_level_jurisdiction_name(self, tmp_path, mock_cli_args):
        mock_cli_args(
            [
                "--state",
                "CA",
                "--code-slug",
                "penal-code",
                "--name",
                "California Penal Code",
            ]
        )
        j_parquet = tmp_path / "jurisdictions.parquet"
        c_parquet = tmp_path / "codes.parquet"

        with (
            patch.object(pipeline_init, "JURISDICTIONS_PARQUET", j_parquet),
            patch.object(pipeline_init, "CODES_PARQUET", c_parquet),
            patch(
                "legiscope.pipeline.init.create_code_structure", return_value=tmp_path
            ),
        ):
            pipeline_init.main()

        j_df = pl.read_parquet(j_parquet)
        assert j_df["name"][0] == "CA"
        assert j_df["level"][0] == "state"
