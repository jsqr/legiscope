"""
Tests for utility configuration classes (LLMConfig, etc.)
"""

import pytest
from unittest.mock import Mock

from legiscope.utils import LLMConfig


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with just required parameters."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client)

        assert config.client is mock_client
        assert config.model is not None  # Should be set by __post_init__
        assert config.temperature == 0.0  # Default
        assert config.max_retries == 3  # Default

    def test_explicit_model(self):
        """Test config with explicit model specified."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, model="gpt-4")

        assert config.model == "gpt-4"

    def test_custom_temperature(self):
        """Test config with custom temperature."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, temperature=0.5)

        assert config.temperature == 0.5

    def test_custom_max_retries(self):
        """Test config with custom max_retries."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, max_retries=5)

        assert config.max_retries == 5

    def test_all_custom_params(self):
        """Test config with all parameters customized."""
        mock_client = Mock()
        config = LLMConfig(
            client=mock_client, model="gpt-4-turbo", temperature=0.7, max_retries=10
        )

        assert config.client is mock_client
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.7
        assert config.max_retries == 10

    def test_temperature_validation_too_low(self):
        """Test that temperature below 0 raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(client=mock_client, temperature=-0.1)

    def test_temperature_validation_too_high(self):
        """Test that temperature above 2.0 raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(client=mock_client, temperature=2.1)

    def test_temperature_at_boundaries(self):
        """Test that boundary values for temperature are accepted."""
        mock_client = Mock()

        # Lower boundary
        config_low = LLMConfig(client=mock_client, temperature=0.0)
        assert config_low.temperature == 0.0

        # Upper boundary
        config_high = LLMConfig(client=mock_client, temperature=2.0)
        assert config_high.temperature == 2.0

    def test_max_retries_validation(self):
        """Test that negative max_retries raises error."""
        mock_client = Mock()
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LLMConfig(client=mock_client, max_retries=-1)

    def test_max_retries_zero(self):
        """Test that zero max_retries is allowed."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, max_retries=0)
        assert config.max_retries == 0

    def test_config_is_dataclass(self):
        """Test that LLMConfig behaves as a dataclass."""
        mock_client = Mock()
        config1 = LLMConfig(client=mock_client, model="gpt-4")
        config2 = LLMConfig(client=mock_client, model="gpt-4")

        # Dataclasses support equality comparison
        assert config1.model == config2.model
        assert config1.temperature == config2.temperature

    def test_config_repr(self):
        """Test that config has useful repr."""
        mock_client = Mock()
        config = LLMConfig(client=mock_client, model="test-model")
        repr_str = repr(config)

        assert "LLMConfig" in repr_str
        assert "test-model" in repr_str


class TestRetrievalConfig:
    """Test RetrievalSettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.retrieve import RetrievalSettings

        settings = RetrievalSettings()

        assert settings.n_results == 10  # Default
        assert settings.jurisdiction_id is None
        assert settings.use_hyde is False

    def test_with_jurisdiction(self):
        """Test settings with jurisdiction filter."""
        from legiscope.retrieve import RetrievalSettings

        settings = RetrievalSettings(jurisdiction_id="IL-WindyCity")

        assert settings.jurisdiction_id == "IL-WindyCity"

    def test_with_hyde(self):
        """Test settings with HYDE rewriting enabled."""
        from legiscope.retrieve import RetrievalSettings
        from unittest.mock import Mock

        mock_client = Mock()
        settings = RetrievalSettings(use_hyde=True, hyde_client=mock_client)

        assert settings.use_hyde is True
        assert settings.hyde_client is mock_client

    def test_hyde_without_client_raises_error(self):
        """Test that use_hyde=True without hyde_client raises error."""
        from legiscope.retrieve import RetrievalSettings

        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalSettings(use_hyde=True)

    def test_empty_query_text_raises_error(self):
        """Test that empty query_text is validated at function call."""
        from legiscope.retrieve import retrieve_segments
        from unittest.mock import Mock

        # query_text validation moved to function, not settings
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            retrieve_segments(
                Mock(),
                "",  # Empty query_text
            )

    def test_invalid_n_results_raises_error(self):
        """Test that invalid n_results raises error."""
        from legiscope.retrieve import RetrievalSettings

        with pytest.raises(ValueError, match="n_results must be positive"):
            RetrievalSettings(n_results=0)


class TestSectionRetrievalConfig:
    """Test SectionRetrievalSettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.retrieve import SectionRetrievalSettings

        settings = SectionRetrievalSettings()

        # All inherited from RetrievalSettings
        assert settings.n_results == 10
        assert settings.use_hyde is False

    def test_missing_parquet_path_raises_error(self):
        """Test that sections_parquet_path is now a function parameter."""
        from legiscope.retrieve import retrieve_sections
        from unittest.mock import Mock

        # sections_parquet_path is now a required function parameter
        with pytest.raises(TypeError):
            retrieve_sections(
                Mock(),
                # Missing sections_parquet_path
                "test query",
            )

    def test_inherits_from_retrieval_config(self):
        """Test that SectionRetrievalSettings inherits RetrievalSettings attributes."""
        from legiscope.retrieve import SectionRetrievalSettings
        from unittest.mock import Mock

        mock_client = Mock()
        settings = SectionRetrievalSettings(
            jurisdiction_id="IL-WindyCity",
            n_results=20,
            use_hyde=True,
            hyde_client=mock_client,
        )

        # Check inherited attributes work
        assert settings.jurisdiction_id == "IL-WindyCity"
        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.hyde_client is mock_client


class TestQueryConfig:
    """Test QueryConfig dataclass."""

    def test_minimal_config(self):
        """Test creating settings with required parameters."""
        from legiscope.query import QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        llm_config = LLMConfig(client=Mock())
        settings = QuerySettings(llm=llm_config)

        assert settings.llm is llm_config
        assert settings.filter_relevance is False
        assert settings.relevance_threshold == 0.5

    def test_with_filtering(self):
        """Test settings with relevance filtering enabled."""
        from legiscope.query import QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        llm_config = LLMConfig(client=Mock())
        settings = QuerySettings(
            llm=llm_config, filter_relevance=True, relevance_threshold=0.7
        )

        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.7
        assert settings.filter_llm is llm_config  # Should use same LLM

    def test_with_separate_filter_llm(self):
        """Test settings with separate LLM for filtering."""
        from legiscope.query import QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        main_llm = LLMConfig(client=Mock(), model="gpt-4")
        filter_llm = LLMConfig(client=Mock(), model="gpt-3.5")

        settings = QuerySettings(
            llm=main_llm, filter_relevance=True, filter_llm=filter_llm
        )

        assert settings.filter_llm is filter_llm
        assert settings.filter_llm is not main_llm

    def test_empty_query_raises_error(self):
        """Test that empty query is validated at function call."""
        from legiscope.query import query_legal_documents, QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        # query validation moved to function, not settings
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        with pytest.raises(ValueError, match="query cannot be empty"):
            query_legal_documents(
                {"sections": []},
                "",  # Empty query
                settings,
            )

    def test_empty_results_raises_error(self):
        """Test that empty retrieval_results is validated at function call."""
        from legiscope.query import query_legal_documents, QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        # retrieval_results validation moved to function, not settings
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        with pytest.raises(ValueError, match="retrieval_results cannot be empty"):
            query_legal_documents(
                None,  # Empty results
                "test",
                settings,
            )

    def test_invalid_relevance_threshold(self):
        """Test that invalid relevance_threshold raises error."""
        from legiscope.query import QuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        with pytest.raises(ValueError, match="relevance_threshold must be between"):
            QuerySettings(llm=LLMConfig(client=Mock()), relevance_threshold=1.5)


class TestBatchQueryConfig:
    """Test BatchQuerySettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.query import BatchQuerySettings
        from unittest.mock import Mock, patch

        # Mock the API client creation to avoid needing API keys,
        # but still test that __post_init__ creates default LLM config
        with patch("legiscope.llm_config.Config.get_fast_client") as mock_get_client:
            mock_get_client.return_value = Mock()

            settings = BatchQuerySettings()

            assert settings.llm is not None  # Should be set by __post_init__
            assert settings.n_results == 10  # Default
            assert settings.use_hyde is False
            mock_get_client.assert_called_once()  # Verify default behavior triggered

    def test_with_custom_llm(self):
        """Test settings with custom LLM."""
        from legiscope.query import BatchQuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        llm_config = LLMConfig(client=Mock(), model="gpt-4")
        settings = BatchQuerySettings(llm=llm_config)

        assert settings.llm is llm_config

    def test_with_all_options(self):
        """Test settings with all options customized."""
        from legiscope.query import BatchQuerySettings
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock

        llm_config = LLMConfig(client=Mock())
        settings = BatchQuerySettings(
            llm=llm_config,
            n_results=20,
            use_hyde=True,
            filter_relevance=True,
            relevance_threshold=0.8,
        )

        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.8

    def test_empty_queries_raises_error(self):
        """Test that empty queries list is validated at function call."""
        from legiscope.query import run_queries
        from unittest.mock import Mock

        # queries validation moved to function, not settings
        with pytest.raises(ValueError, match="queries list cannot be empty"):
            run_queries(
                collection=Mock(),
                sections_parquet_path="./data/sections.parquet",
                queries=[],  # Empty queries
                jurisdiction_id="IL-WindyCity",
            )

    def test_empty_jurisdiction_raises_error(self):
        """Test that empty jurisdiction_id is validated at function call."""
        from legiscope.query import run_queries
        from unittest.mock import Mock

        # jurisdiction_id validation moved to function, not settings
        with pytest.raises(ValueError, match="jurisdiction_id cannot be empty"):
            run_queries(
                collection=Mock(),
                sections_parquet_path="./data/sections.parquet",
                queries=["test"],
                jurisdiction_id="",  # Empty jurisdiction_id
            )

    def test_invalid_n_results(self):
        """Test that invalid n_results raises error."""
        from legiscope.query import BatchQuerySettings

        with pytest.raises(ValueError, match="n_results must be positive"):
            BatchQuerySettings(n_results=0)
