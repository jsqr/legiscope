"""
Legiscope - Automated analysis of municipal codes for legal epidemiology.

This package provides tools for:
- Converting legal text to structured markdown
- Segmenting legal code into manageable chunks
- Creating embeddings for semantic search
- Retrieving relevant legal passages using ChromaDB
"""

# Core functionality
from legiscope.parse.convert import text2md
from legiscope.parse.headings import BooleanResult, HeadingLevel, HeadingStructure
from legiscope.parse.scan import scan_legal_text
from legiscope.utils import ask
from legiscope.embeddings import (
    EmbeddingIndexConfig,
    create_and_save_embeddings,
    create_embedding_index,
    get_embeddings,
    get_or_create_legal_collection,
)
from legiscope.models import (
    CodeRef,
    JurisdictionRef,
)
from legiscope.query import (
    BatchQuerySettings,
    LegalQueryResponse,
    QuerySettings,
    query_legal_documents,
    run_queries,
)
from legiscope.retrieve import (
    RetrievalSettings,
    SectionRetrievalSettings,
    get_jurisdiction_stats,
    hyde_rewriter,
    retrieve_sections,
    retrieve_segments,
)
from legiscope.segment import (
    add_parent_relationships,
    create_segments_df,
    divide_into_sections,
    enrich_sections,
    get_section_text,
    segment_text,
)

# Version
__version__ = "0.1.0"
__all__ = [
    # Models module
    "JurisdictionRef",
    "CodeRef",
    # Convert module
    "scan_legal_text",
    "text2md",
    "ask",
    "BooleanResult",
    "HeadingLevel",
    "HeadingStructure",
    # Segment module
    "divide_into_sections",
    "add_parent_relationships",
    "enrich_sections",
    "get_section_text",
    "segment_text",
    "create_segments_df",
    # Embeddings module
    "get_embeddings",
    "create_embedding_index",
    "EmbeddingIndexConfig",
    "get_or_create_legal_collection",
    "create_and_save_embeddings",
    # Retrieve module
    "retrieve_segments",
    "retrieve_sections",
    "RetrievalSettings",
    "SectionRetrievalSettings",
    "hyde_rewriter",
    "get_jurisdiction_stats",
    # Query module
    "query_legal_documents",
    "run_queries",
    "QuerySettings",
    "BatchQuerySettings",
    "LegalQueryResponse",
]
