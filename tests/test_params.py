"""Tests for legiscope.params — pipeline parameter loader."""

from unittest.mock import patch

import yaml

from legiscope import params


class TestLoadParams:
    def test_loads_global_params(self):
        """Global params.yaml loads and contains expected top-level keys."""
        p = params.load_params()
        assert "llm" in p
        assert "embeddings" in p
        assert "retrieval" in p
        assert "query" in p

    def test_llm_provider_default(self):
        p = params.load_params()
        assert p["llm"]["default_provider"] == "mistral"

    def test_provider_models_present(self):
        p = params.load_params()
        providers = p["llm"]["providers"]
        assert "openai" in providers
        assert "mistral" in providers
        assert "ollama" in providers
        assert providers["mistral"]["fast"] == "mistral-small-2506"


class TestPerCodeMerge:
    def test_per_code_override(self, tmp_path):
        """Per-code params.yaml is deep-merged on top of global params."""
        override = {"llm": {"default_provider": "ollama"}}
        override_file = tmp_path / "params.yaml"
        override_file.write_text(yaml.dump(override))

        p = params.load_params(code_data_dir=tmp_path)
        assert p["llm"]["default_provider"] == "ollama"
        # Other keys should still exist
        assert "providers" in p["llm"]

    def test_no_override_file(self, tmp_path):
        """Missing per-code params.yaml is silently ignored."""
        p = params.load_params(code_data_dir=tmp_path)
        assert "llm" in p


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = params._deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"llm": {"provider": "mistral", "temperature": 0.0}}
        override = {"llm": {"provider": "openai"}}
        result = params._deep_merge(base, override)
        assert result["llm"]["provider"] == "openai"
        assert result["llm"]["temperature"] == 0.0

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        params._deep_merge(base, override)
        assert base["a"]["b"] == 1  # original not mutated


class TestDVCFallback:
    def test_falls_back_to_yaml_when_dvc_unavailable(self):
        """When dvc.api.params_show() fails, loads from file directly."""
        with patch("dvc.api.params_show", side_effect=Exception("no DVC")):
            p = params.load_params()
            assert "llm" in p

    def test_uses_dvc_when_available(self):
        """When dvc.api.params_show() works, uses its output."""
        fake_params = {"llm": {"default_provider": "dvc-test"}}
        with patch("dvc.api.params_show", return_value=fake_params):
            p = params.load_params()
            assert p["llm"]["default_provider"] == "dvc-test"
