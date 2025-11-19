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
        assert config.temperature == 0.1  # Default
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
            client=mock_client,
            model="gpt-4-turbo",
            temperature=0.7,
            max_retries=10
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
    """Test RetrievalConfig dataclass."""
    
    def test_minimal_config(self):
        """Test creating config with just required parameters."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        mock_collection = Mock()
        config = RetrievalConfig(
            collection=mock_collection,
            query_text="test query"
        )
        
        assert config.collection is mock_collection
        assert config.query_text == "test query"
        assert config.n_results == 10  # Default
        assert config.jurisdiction_id is None
        assert config.use_hyde is False
    
    def test_with_jurisdiction(self):
        """Test config with jurisdiction filter."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        config = RetrievalConfig(
            collection=Mock(),
            query_text="test",
            jurisdiction_id="IL-WindyCity"
        )
        
        assert config.jurisdiction_id == "IL-WindyCity"
    
    def test_with_hyde(self):
        """Test config with HYDE rewriting enabled."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        mock_client = Mock()
        config = RetrievalConfig(
            collection=Mock(),
            query_text="test",
            use_hyde=True,
            hyde_client=mock_client
        )
        
        assert config.use_hyde is True
        assert config.hyde_client is mock_client
    
    def test_hyde_without_client_raises_error(self):
        """Test that use_hyde=True without hyde_client raises error."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalConfig(
                collection=Mock(),
                query_text="test",
                use_hyde=True
            )
    
    def test_empty_query_text_raises_error(self):
        """Test that empty query_text raises error."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            RetrievalConfig(
                collection=Mock(),
                query_text=""
            )
    
    def test_invalid_n_results_raises_error(self):
        """Test that invalid n_results raises error."""
        from legiscope.retrieve import RetrievalConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="n_results must be positive"):
            RetrievalConfig(
                collection=Mock(),
                query_text="test",
                n_results=0
            )


class TestSectionRetrievalConfig:
    """Test SectionRetrievalConfig dataclass."""
    
    def test_minimal_config(self):
        """Test creating config with required parameters."""
        from legiscope.retrieve import SectionRetrievalConfig
        from unittest.mock import Mock
        
        config = SectionRetrievalConfig(
            collection=Mock(),
            query_text="test",
            sections_parquet_path="./data/sections.parquet"
        )
        
        assert config.sections_parquet_path == "./data/sections.parquet"
        assert config.query_text == "test"
    
    def test_missing_parquet_path_raises_error(self):
        """Test that missing sections_parquet_path raises error."""
        from legiscope.retrieve import SectionRetrievalConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="sections_parquet_path is required"):
            SectionRetrievalConfig(
                collection=Mock(),
                query_text="test"
            )
    
    def test_inherits_from_retrieval_config(self):
        """Test that SectionRetrievalConfig inherits RetrievalConfig attributes."""
        from legiscope.retrieve import SectionRetrievalConfig
        from unittest.mock import Mock
        
        mock_client = Mock()
        config = SectionRetrievalConfig(
            collection=Mock(),
            query_text="test",
            sections_parquet_path="./data/sections.parquet",
            jurisdiction_id="IL-WindyCity",
            n_results=20,
            use_hyde=True,
            hyde_client=mock_client
        )
        
        # Check inherited attributes work
        assert config.jurisdiction_id == "IL-WindyCity"
        assert config.n_results == 20
        assert config.use_hyde is True
        assert config.hyde_client is mock_client


class TestQueryConfig:
    """Test QueryConfig dataclass."""
    
    def test_minimal_config(self):
        """Test creating config with required parameters."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        llm_config = LLMConfig(client=Mock())
        config = QueryConfig(
            llm=llm_config,
            query="test query",
            retrieval_results={"sections": []}
        )
        
        assert config.llm is llm_config
        assert config.query == "test query"
        assert config.filter_relevance is False
        assert config.relevance_threshold == 0.5
    
    def test_with_filtering(self):
        """Test config with relevance filtering enabled."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        llm_config = LLMConfig(client=Mock())
        config = QueryConfig(
            llm=llm_config,
            query="test",
            retrieval_results={"sections": []},
            filter_relevance=True,
            relevance_threshold=0.7
        )
        
        assert config.filter_relevance is True
        assert config.relevance_threshold == 0.7
        assert config.filter_llm is llm_config  # Should use same LLM
    
    def test_with_separate_filter_llm(self):
        """Test config with separate LLM for filtering."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        main_llm = LLMConfig(client=Mock(), model="gpt-4")
        filter_llm = LLMConfig(client=Mock(), model="gpt-3.5")
        
        config = QueryConfig(
            llm=main_llm,
            query="test",
            retrieval_results={"sections": []},
            filter_relevance=True,
            filter_llm=filter_llm
        )
        
        assert config.filter_llm is filter_llm
        assert config.filter_llm is not main_llm
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises error."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            QueryConfig(
                llm=LLMConfig(client=Mock()),
                query="",
                retrieval_results={"sections": []}
            )
    
    def test_empty_results_raises_error(self):
        """Test that empty retrieval_results raises error."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="retrieval_results cannot be empty"):
            QueryConfig(
                llm=LLMConfig(client=Mock()),
                query="test",
                retrieval_results=None
            )
    
    def test_invalid_relevance_threshold(self):
        """Test that invalid relevance_threshold raises error."""
        from legiscope.query import QueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="relevance_threshold must be between"):
            QueryConfig(
                llm=LLMConfig(client=Mock()),
                query="test",
                retrieval_results={"sections": []},
                relevance_threshold=1.5
            )


class TestBatchQueryConfig:
    """Test BatchQueryConfig dataclass."""
    
    def test_minimal_config(self):
        """Test creating config with required parameters."""
        from legiscope.query import BatchQueryConfig
        from unittest.mock import Mock
        
        config = BatchQueryConfig(
            queries=["query1", "query2"],
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="./data/sections.parquet",
            collection=Mock()
        )
        
        assert config.queries == ["query1", "query2"]
        assert config.jurisdiction_id == "IL-WindyCity"
        assert config.llm is not None  # Should be set by __post_init__
        assert config.n_results == 10  # Default
        assert config.use_hyde is False
    
    def test_with_custom_llm(self):
        """Test config with custom LLM."""
        from legiscope.query import BatchQueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        llm_config = LLMConfig(client=Mock(), model="gpt-4")
        config = BatchQueryConfig(
            queries=["test"],
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="./data/sections.parquet",
            collection=Mock(),
            llm=llm_config
        )
        
        assert config.llm is llm_config
    
    def test_with_all_options(self):
        """Test config with all options customized."""
        from legiscope.query import BatchQueryConfig
        from legiscope.utils import LLMConfig
        from unittest.mock import Mock
        
        llm_config = LLMConfig(client=Mock())
        config = BatchQueryConfig(
            queries=["q1", "q2"],
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="./data/sections.parquet",
            collection=Mock(),
            llm=llm_config,
            n_results=20,
            use_hyde=True,
            filter_relevance=True,
            relevance_threshold=0.8
        )
        
        assert config.n_results == 20
        assert config.use_hyde is True
        assert config.filter_relevance is True
        assert config.relevance_threshold == 0.8
    
    def test_empty_queries_raises_error(self):
        """Test that empty queries list raises error."""
        from legiscope.query import BatchQueryConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="queries list cannot be empty"):
            BatchQueryConfig(
                queries=[],
                jurisdiction_id="IL-WindyCity",
                sections_parquet_path="./data/sections.parquet",
                collection=Mock()
            )
    
    def test_empty_jurisdiction_raises_error(self):
        """Test that empty jurisdiction_id raises error."""
        from legiscope.query import BatchQueryConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="jurisdiction_id cannot be empty"):
            BatchQueryConfig(
                queries=["test"],
                jurisdiction_id="",
                sections_parquet_path="./data/sections.parquet",
                collection=Mock()
            )
    
    def test_invalid_n_results(self):
        """Test that invalid n_results raises error."""
        from legiscope.query import BatchQueryConfig
        from unittest.mock import Mock
        
        with pytest.raises(ValueError, match="n_results must be positive"):
            BatchQueryConfig(
                queries=["test"],
                jurisdiction_id="IL-WindyCity",
                sections_parquet_path="./data/sections.parquet",
                collection=Mock(),
                n_results=0
            )
