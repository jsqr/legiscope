"""
Legiscope - Automated analysis of municipal codes for legal epidemiology.

This package provides tools for:
- Converting legal text to structured markdown
- Segmenting legal code into manageable chunks
- Creating embeddings for semantic search
- Retrieving relevant legal passages using ChromaDB
"""

# Core functionality
from legiscope.convert import (
    scan_legal_text,
    text2md,
    ask,
    BooleanResult,
    HeadingLevel,
    HeadingStructure,
)

from legiscope.segment import (
    divide_into_sections,
    add_parent_relationships,
    segment_text,
    add_segments_to_sections,
    create_segments_df,
)

from legiscope.embeddings import (
    get_embeddings,
    create_embeddings_df,
    create_embedding_index,
    get_or_create_legal_collection,
    add_jurisdiction_embeddings,
    create_and_persist_embeddings,
)

from legiscope.retrieve import (
    retrieve_embeddings,
    retrieve_sections,
    hyde_rewriter,
    get_jurisdiction_stats,
    compare_jurisdictions,
)


# Version
__version__ = "0.1.0"
__all__ = [
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
    "segment_text",
    "add_segments_to_sections",
    "create_segments_df",
    # Embeddings module
    "get_embeddings",
    "create_embeddings_df",
    "create_embedding_index",
    "get_or_create_legal_collection",
    "add_jurisdiction_embeddings",
    "create_and_persist_embeddings",
    # Retrieve module
    "retrieve_embeddings",
    "retrieve_sections",
    "hyde_rewriter",
    "get_jurisdiction_stats",
    "compare_jurisdictions",
]
