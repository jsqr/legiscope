"""Tests for legiscope.models — JurisdictionRef, CodeRef, and schema constants."""

from pathlib import Path

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
        assert ref.municipality is None

    def test_state_and_municipality(self):
        ref = JurisdictionRef(state="CA", municipality="LosAngeles")
        assert ref.jurisdiction_id == "CA-LosAngeles"
        assert ref.level == "local"
        assert ref.municipality == "LosAngeles"

    def test_state_normalised_to_uppercase(self):
        ref = JurisdictionRef(state="ca")
        assert ref.state == "CA"
        assert ref.jurisdiction_id == "CA"

    def test_state_whitespace_stripped(self):
        ref = JurisdictionRef(state="  il  ")
        assert ref.state == "IL"

    def test_municipality_spaces_removed(self):
        ref = JurisdictionRef(state="CA", municipality="Los Angeles")
        assert ref.municipality == "LosAngeles"
        assert ref.jurisdiction_id == "CA-LosAngeles"

    def test_empty_state_raises(self):
        with pytest.raises(ValueError, match="state cannot be empty"):
            JurisdictionRef(state="")

    def test_whitespace_only_state_raises(self):
        with pytest.raises(ValueError, match="state cannot be empty"):
            JurisdictionRef(state="   ")

    def test_empty_municipality_string_raises(self):
        with pytest.raises(ValueError, match="municipality cannot be empty"):
            JurisdictionRef(state="CA", municipality="  ")

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
        j = JurisdictionRef(state="CA", municipality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.code_id == "CA:LosAngeles:municipal-code"

    def test_jurisdiction_id_shortcut(self):
        j = JurisdictionRef(state="IL", municipality="Chicago")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.jurisdiction_id == "IL-Chicago"

    def test_section_id(self):
        j = JurisdictionRef(state="CA", municipality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.section_id(42) == "CA:LosAngeles:municipal-code:s42"

    def test_segment_id(self):
        j = JurisdictionRef(state="CA", municipality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.segment_id(7) == "CA:LosAngeles:municipal-code:g7"

    def test_data_dir_state_level(self):
        j = JurisdictionRef(state="CA")
        c = CodeRef(jurisdiction=j, code_slug="penal-code")
        assert c.data_dir == Path("CA/State/penal-code")

    def test_data_dir_local(self):
        j = JurisdictionRef(state="CA", municipality="LosAngeles")
        c = CodeRef(jurisdiction=j, code_slug="municipal-code")
        assert c.data_dir == Path("CA/LosAngeles/municipal-code")

    def test_full_data_dir(self):
        j = JurisdictionRef(state="IL", municipality="Chicago")
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


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_jurisdictions_schema_keys(self):
        expected = {
            "jurisdiction_id",
            "state",
            "municipality",
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
