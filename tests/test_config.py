"""Tests for legiscope.config — infrastructure configuration loader."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from legiscope import config as cfg


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset cached config before and after every test."""
    cfg.reset()
    yield
    cfg.reset()


class TestConfigLoading:
    def test_loads_config_yaml(self):
        """Config loads without error and returns a dict."""
        val = cfg.get("paths.data_dir")
        assert val is not None

    def test_dot_path_access(self):
        assert cfg.get("paths.data_dir") == "data"
        assert cfg.get("paths.laws_dir") == "laws"

    def test_nested_dot_path(self):
        assert cfg.get("database.chromadb.default_collection") == "legal_code"

    def test_missing_key_returns_default(self):
        assert cfg.get("nonexistent.key") is None
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_partial_path_returns_dict(self):
        result = cfg.get("paths")
        assert isinstance(result, dict)
        assert "data_dir" in result


class TestDataDirOverride:
    def test_default_data_dir(self):
        assert cfg.data_dir() == Path("data")

    def test_env_var_override(self):
        with patch.dict(os.environ, {"LEGISCOPE_DATA_DIR": "/tmp/custom_data"}):
            assert cfg.data_dir() == Path("/tmp/custom_data")

    def test_laws_dir_follows_data_dir(self):
        with patch.dict(os.environ, {"LEGISCOPE_DATA_DIR": "/tmp/custom_data"}):
            assert cfg.laws_dir() == Path("/tmp/custom_data/laws")

    def test_chroma_db_path_follows_data_dir(self):
        with patch.dict(os.environ, {"LEGISCOPE_DATA_DIR": "/tmp/custom_data"}):
            assert cfg.chroma_db_path() == Path("/tmp/custom_data/chroma_db")


class TestProperties:
    def test_laws_dir(self):
        assert cfg.laws_dir() == Path("data/laws")

    def test_chroma_db_path(self):
        assert cfg.chroma_db_path() == Path("data/chroma_db")

    def test_queries_dir(self):
        assert cfg.queries_dir() == Path("data/queries")

    def test_output_dir(self):
        assert cfg.output_dir() == Path("data/output")

    def test_default_queries_path(self):
        assert cfg.default_queries_path() == Path(
            "data/queries/drug_paraphernalia_queries_clean.csv"
        )

    def test_monqcle_report_path(self):
        assert cfg.monqcle_report_path() == Path(
            "coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv"
        )

    def test_queries_dir_follows_data_dir(self):
        with patch.dict(os.environ, {"LEGISCOPE_DATA_DIR": "/tmp/custom_data"}):
            assert cfg.queries_dir() == Path("/tmp/custom_data/queries")

    def test_output_dir_follows_data_dir(self):
        with patch.dict(os.environ, {"LEGISCOPE_DATA_DIR": "/tmp/custom_data"}):
            assert cfg.output_dir() == Path("/tmp/custom_data/output")


class TestMissingFile:
    def test_missing_config_raises(self, tmp_path, monkeypatch):
        """FileNotFoundError when config.yaml cannot be found."""
        # Point the module search to an empty tree
        monkeypatch.setattr(
            cfg,
            "_find_config_path",
            lambda: (_ for _ in ()).throw(
                FileNotFoundError("Could not find config.yaml")
            ),
        )
        with pytest.raises(FileNotFoundError):
            cfg._load()
