"""
Tests for new BatchQuerySettings parameters.
"""

import pytest
from unittest.mock import Mock
from legiscope.utils import LLMConfig
from legiscope.query import BatchQuerySettings

def test_batch_query_settings_defaults():
    """Test default values for new parameters."""
    mock_llm = Mock(spec=LLMConfig)
    settings = BatchQuerySettings(llm=mock_llm)
    
    assert settings.n_results == 10
    assert settings.use_hyde is False
    assert settings.filter_relevance is False
    assert settings.relevance_threshold == 0.5
    assert settings.validate_supporting_passages is True

def test_batch_query_settings_instantiation():
    """Test instantiating with specific values."""
    mock_llm = Mock(spec=LLMConfig)
    settings = BatchQuerySettings(
        llm=mock_llm,
        n_results=20,
        use_hyde=True,
        filter_relevance=True,
        relevance_threshold=0.8,
        validate_supporting_passages=False
    )
    
    assert settings.n_results == 20
    assert settings.use_hyde is True
    assert settings.filter_relevance is True
    assert settings.relevance_threshold == 0.8
    assert settings.validate_supporting_passages is False
