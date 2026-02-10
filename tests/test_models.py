"""Tests for legiscope.models — JurisdictionRef, CodeRef, and schema constants."""

from pathlib import Path
from unittest.mock import patch

import pytest

from legiscope.models import (
    CODES_PARQUET,
    CODES_SCHEMA,
    EXTERNAL_REFERENCES_SCHEMA,
    JURISDICTIONS_PARQUET,
    JURISDICTIONS_SCHEMA,
    LAWS_DIR,
    RELATIONS_SCHEMA,
    CodeRef,
    JurisdictionRef,
)

# ---------------------------------------------------------------------------
# JurisdictionRef
# ---------------------------------------------------------------------------


class TestJurisdictionRef:
    def test_state_only(self):
        ref = JurisdictionRef(state="CA")
        assert ref.jurisdiction_id == "CA"
        assert ref.level == "state"
        assert ref.locality is None

    def test_state_and_locality(self):
        ref = JurisdictionRef(state="CA", locality="LosAngeles")
        assert ref.jurisdiction_id == "CA-LosAngeles"
        assert ref.level == "local"
        assert ref.locality == "LosAngeles"

    def test_state_normalised_to_uppercase(self):
        ref = JurisdictionRef(state="ca")
        assert ref.state == "CA"
        assert ref.jurisdiction_id == "CA"

    def test_state_whitespace_stripped(self):
        ref = JurisdictionRef(state="  il  ")
        assert ref.state == "IL"

    def test_locality_spaces_removed(self):
        ref = JurisdictionRef(state="CA", locality="Los Angeles")
        assert ref.locality == "LosAngeles"
        assert ref.jurisdiction_id == "CA-LosAngeles"

    def test_empty_state_raises(self):
        with pytest.raises(ValueError, match="state cannot be empty"):
            JurisdictionRef(state="")

    def test_whitespace_only_state_raises(self):
        with pytest.raises(ValueError, match="state cannot be empty"):
            JurisdictionRef(state="   ")

    def test_empty_locality_string_raises(self):
        with pytest.raises(ValueError, match="locality cannot be empty"):
            JurisdictionRef(state="CA", locality="  ")

    def test_frozen(self):
        ref = JurisdictionRef(state="CA")
        with pytest.raises(AttributeError):
            ref.state = "NY"


# ---------------------------------------------------------------------------
# CodeRef
# ---------------------------------------------------------------------------


class TestCodeRef:
    def test_state_level_code_id(self):
        j = JurisdictionRef(state="CA")
        c = CodeRef(jurisdiction=j, code_slug="penal-code")
        assert c.code_id == "CA:state:penal-code"

    def test_local_code_id(self):
        j = JurisdictionRef(state="CA", locality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.code_id == "CA:LosAngeles:municipal-code"

    def test_jurisdiction_id_shortcut(self):
        j = JurisdictionRef(state="IL", locality="Chicago")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.jurisdiction_id == "IL-Chicago"

    def test_section_id(self):
        j = JurisdictionRef(state="CA", locality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.section_id(42) == "CA:LosAngeles:municipal-code:s42"

    def test_segment_id(self):
        j = JurisdictionRef(state="CA", locality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.segment_id(7) == "CA:LosAngeles:municipal-code:g7"

    def test_data_dir_state_level(self):
        j = JurisdictionRef(state="CA")
        c = CodeRef(jurisdiction=j, code_slug="penal-code")
        assert c.data_dir == Path("CA/State/penal-code")

    def test_data_dir_local(self):
        j = JurisdictionRef(state="CA", locality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.data_dir == Path("CA/LosAngeles/municipal-code")

    def test_full_data_dir(self):
        j = JurisdictionRef(state="IL", locality="Chicago")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.full_data_dir == LAWS_DIR / "IL" / "Chicago" / "municipal-code"

    def test_empty_code_slug_raises(self):
        j = JurisdictionRef(state="CA")
        with pytest.raises(ValueError, match="code_slug cannot be empty"):
            CodeRef(jurisdiction=j, code_slug="")

    def test_whitespace_code_slug_raises(self):
        j = JurisdictionRef(state="CA")
        with pytest.raises(ValueError, match="code_slug cannot be empty"):
            CodeRef(jurisdiction=j, code_slug="   ")

    def test_code_slug_stripped(self):
        j = JurisdictionRef(state="CA")
        c = CodeRef(jurisdiction=j, code_slug="  penal-code  ")
        assert c.code_slug == "penal-code"

    def test_frozen(self):
        j = JurisdictionRef(state="CA")
        c = CodeRef(jurisdiction=j, code_slug="penal-code")
        with pytest.raises(AttributeError):
            c.code_slug = "other"


class TestCodeRefFromDvcVars:
    def test_basic_local(self):
        ref = CodeRef.from_dvc_vars(state="CA", locality="LosAngeles", code_slug="mc")
        assert ref.jurisdiction.state == "CA"
        assert ref.jurisdiction.locality == "LosAngeles"
        assert ref.jurisdiction.level == "local"

    def test_state_sentinel_normalised_to_none(self):
        """'State' locality sentinel is normalised to None."""
        ref = CodeRef.from_dvc_vars(
            state="CA", locality="State", code_slug="penal-code"
        )
        assert ref.jurisdiction.locality is None
        assert ref.jurisdiction.level == "state"
        assert ref.jurisdiction_id == "CA"
        assert str(ref.data_dir) == "CA/State/penal-code"

    def test_state_sentinel_with_whitespace(self):
        ref = CodeRef.from_dvc_vars(state="CA", locality="  State  ", code_slug="pc")
        assert ref.jurisdiction.locality is None
        assert ref.jurisdiction.level == "state"

    def test_none_locality(self):
        ref = CodeRef.from_dvc_vars(state="CA", locality=None, code_slug="pc")
        assert ref.jurisdiction.locality is None
        assert ref.jurisdiction.level == "state"

    def test_missing_state_raises(self):
        with pytest.raises(ValueError, match="state is required"):
            CodeRef.from_dvc_vars(state=None, code_slug="pc")

    def test_missing_code_slug_raises(self):
        with pytest.raises(ValueError, match="code_slug is required"):
            CodeRef.from_dvc_vars(state="CA", code_slug=None)


class TestCodeRefFromParams:
    def test_reads_from_params_dict(self):
        params = {
            "jurisdiction": {
                "state": "CA",
                "locality": "LosAngeles",
                "code_slug": "municipal-code",
            }
        }
        ref = CodeRef.from_params(params)
        assert ref.jurisdiction.state == "CA"
        assert ref.jurisdiction.locality == "LosAngeles"
        assert ref.code_slug == "municipal-code"

    def test_state_level(self):
        params = {
            "jurisdiction": {
                "state": "CA",
                "locality": "State",
                "code_slug": "penal-code",
            }
        }
        ref = CodeRef.from_params(params)
        assert ref.jurisdiction.locality is None
        assert ref.jurisdiction.level == "state"

    def test_none_locality(self):
        params = {
            "jurisdiction": {
                "state": "CA",
                "code_slug": "penal-code",
            }
        }
        ref = CodeRef.from_params(params)
        assert ref.jurisdiction.locality is None
        assert ref.jurisdiction.level == "state"

    def test_missing_state_raises(self):
        params = {"jurisdiction": {"code_slug": "mc"}}
        with pytest.raises(ValueError, match="state is required"):
            CodeRef.from_params(params)

    def test_missing_code_slug_raises(self):
        params = {"jurisdiction": {"state": "CA"}}
        with pytest.raises(ValueError, match="code_slug is required"):
            CodeRef.from_params(params)

    def test_loads_params_when_none(self):
        fake_params = {
            "jurisdiction": {
                "state": "TX",
                "locality": "Austin",
                "code_slug": "city-code",
            }
        }
        with patch("legiscope.params.load_params", return_value=fake_params):
            ref = CodeRef.from_params()
        assert ref.jurisdiction.state == "TX"
        assert ref.jurisdiction.locality == "Austin"
        assert ref.code_slug == "city-code"


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_jurisdictions_schema_keys(self):
        expected = {
            "jurisdiction_id",
            "state",
            "locality",
            "level",
            "name",
            "parent_jurisdiction",
        }
        assert set(JURISDICTIONS_SCHEMA.keys()) == expected

    def test_codes_schema_keys(self):
        expected = {
            "code_id",
            "jurisdiction_id",
            "code_slug",
            "name",
            "code_type",
            "level",
        }
        assert set(CODES_SCHEMA.keys()) == expected

    def test_relations_schema_keys(self):
        expected = {
            "relation_id",
            "code_id",
            "source_section_id",
            "target_section_id",
            "relation_type",
            "target_text",
            "scope",
            "confidence",
        }
        assert set(RELATIONS_SCHEMA.keys()) == expected

    def test_external_references_schema_keys(self):
        expected = {
            "reference_id",
            "code_id",
            "source_section_id",
            "target_jurisdiction",
            "target_code",
            "target_citation",
            "target_section_id",
            "citation_format",
            "relation_type",
            "confidence",
        }
        assert set(EXTERNAL_REFERENCES_SCHEMA.keys()) == expected


# ---------------------------------------------------------------------------
# Registry paths
# ---------------------------------------------------------------------------


class TestRegistryPaths:
    def test_jurisdictions_parquet_path(self):
        assert JURISDICTIONS_PARQUET == Path("data/jurisdictions.parquet")

    def test_codes_parquet_path(self):
        assert CODES_PARQUET == Path("data/codes.parquet")
