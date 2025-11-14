from pathlib import Path
from typing import Any, cast

import chromadb
import polars as pl
from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.embeddings import get_embeddings, get_embedding_client
from legiscope.llm_config import Config
from legiscope.utils import ask


class HydeRewrite(BaseModel):
    """Structured response for HYDE query rewriting."""

    rewritten_query: str = Field(
        description="The query rewritten in municipal code style for semantic search"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the rewrite quality", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the rewrite approach and key changes made"
    )
    query_type: str = Field(
        description="Type of legal query (e.g., 'zoning', 'permit', 'licensing')"
    )


class RelevanceAssessment(BaseModel):
    """Structured response for relevance assessment of text to a query."""

    is_relevant: bool = Field(
        description="Whether the text is directly relevant to answering the query"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the relevance assessment", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Explanation of why the text is or is not relevant to the query"
    )


def hyde_rewriter(
    query: str, client: Instructor, model: str | None = None
) -> HydeRewrite:
    """Rewrite a natural language query into municipal code style text using HYDE approach.

    Transforms user queries into the style and format of municipal code text to improve
    semantic similarity matching with embedded legal documents using LLM-powered transformation.

    Args:
        query: Natural language query from user
        client: Instructor client for LLM-powered rewriting
        model: LLM model to use. Uses Config.get_fast_model() if not specified

    Returns:
        HydeRewrite: Structured response with rewritten query and metadata

    Raises:
        ValueError: If query is empty or client is invalid

    Example:
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        result = hyde_rewriter("where can I park my car", client)
        print(result.rewritten_query)
        print(result.confidence)
        print(result.query_type)
    """
    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    if not query or not query.strip():
        logger.error("Query cannot be empty for HYDE rewriting")
        raise ValueError("Query cannot be empty for HYDE rewriting")

    if client is None:
        logger.error("Client is required for HYDE rewriting")
        raise ValueError("Client is required for HYDE rewriting")

    logger.info(f"Using LLM for HYDE rewrite: '{query[:50]}...'")

    system_prompt = """You are an expert in municipal law and legal drafting.
Transform the given natural language query into the style and format of municipal
code text to improve semantic search matching against legal documents.

The rewritten query should:
1. Use formal legal language and terminology typical of municipal codes
2. Reference typical municipal code structure and phrasing
3. Maintain the original query's core intent and meaning
4. Be suitable for semantic similarity search against legal documents
5. Be concise but comprehensive enough for effective matching

Common municipal code patterns:
- "The following provisions regulate [topic] within municipal boundaries."
- "This section establishes requirements for [topic]."
- "Regulations concerning [topic] are outlined below."
- "The municipal code addresses [topic] as follows:"
- "The following rules apply to [topic]:"

Classify the query type (e.g., zoning, permits, licensing, noise, animals, etc.)
and provide a confidence score for the rewrite quality."""

    user_prompt = f"""Rewrite the following natural language query into municipal code style:

Original query: "{query}"

Provide a rewritten query that would be effective for semantic search against municipal code documents."""

    try:
        result = ask(
            client=client,
            prompt=user_prompt,
            response_model=HydeRewrite,
            system=system_prompt,
            model=model,
            temperature=0.1,  # Low temperature for consistent legal style
            max_retries=3,
        )

        logger.info(
            f"LLM HYDE rewrite completed - confidence: {result.confidence:.2f}, "
            f"type: {result.query_type}, original: '{query[:30]}...', "
            f"rewritten: '{result.rewritten_query[:30]}...'"
        )

        return result

    except Exception as e:
        logger.error(f"LLM HYDE rewrite failed: {str(e)}")
        raise


def is_relevant(
    client: Instructor, query: str, text: str, model: str | None = None
) -> RelevanceAssessment:
    """Assess whether text is directly relevant to answering a query using LLM analysis.

    Uses LLM-powered analysis to determine if the given text directly helps answer
    the query, providing a structured assessment with confidence score and reasoning.

    Args:
        query: The query being answered
        text: The text to assess for relevance
        client: Instructor client for LLM-powered analysis
        model: LLM model to use. Uses Config.get_fast_model() if not specified

    Returns:
        RelevanceAssessment: Structured assessment with relevance determination

    Raises:
        ValueError: If query or text is empty, or client is invalid

    Example:
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        result = is_relevant(
            "parking regulations",
            "No vehicle shall be parked on any street between 2 AM and 6 AM",
            client
        )
        print(result.is_relevant)
        print(result.confidence)
        print(result.reasoning)
    """
    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    if not query or not query.strip():
        logger.error("Query cannot be empty for relevance assessment")
        raise ValueError("Query cannot be empty for relevance assessment")

    if not text or not text.strip():
        logger.error("Text cannot be empty for relevance assessment")
        raise ValueError("Text cannot be empty for relevance assessment")

    if client is None:
        logger.error("Client is required for relevance assessment")
        raise ValueError("Client is required for relevance assessment")

    logger.info(
        f"Using LLM for relevance assessment: query '{query[:30]}...', text '{text[:30]}...'"
    )

    system_prompt = """You are an expert legal analyst. Determine whether the given text
is directly relevant to answering the query.

The text is considered relevant if it:
1. Directly addresses the query topic
2. Contains specific information that helps answer the query
3. Provides rules, regulations, or guidance related to the query
4. Is not merely tangentially related but substantially useful

The text is NOT relevant if it:
1. Discusses unrelated topics
2. Is too general or vague to be useful
3. Mentions the topic but provides no actionable information
4. Is administrative or procedural content unrelated to the query substance

Provide a confidence score (0-1) indicating how certain you are of the assessment."""

    user_prompt = f"""Assess whether the following text is directly relevant to answering the query:

Query: "{query}"

Text to assess:

"{text}"

Determine if this text directly helps answer the query and provide your assessment with confidence."""

    try:
        result = ask(
            client=client,
            prompt=user_prompt,
            response_model=RelevanceAssessment,
            system=system_prompt,
            model=model,
            temperature=0.1,  # Low temperature
            max_retries=3,
        )

        logger.info(
            f"LLM relevance assessment completed - relevant: {result.is_relevant}, "
            f"confidence: {result.confidence:.2f}, query: '{query[:20]}...', "
            f"text: '{text[:20]}...'"
        )

        return result

    except Exception as e:
        logger.error(f"LLM relevance assessment failed: {str(e)}")
        raise


def filter_results(
    client: Instructor,
    results: dict[str, Any],
    query: str,
    threshold: float = 0.5,
    model: str | None = None,
) -> dict[str, Any]:
    """Filter retrieval results by relevance using LLM-powered assessment.

    Applies relevance assessment to each document in retrieval results and filters
    out documents that are not relevant or fall below the confidence threshold.

    Args:
        results: Retrieval results from retrieve_embeddings or similar functions
        query: Original query used for retrieval
        client: Instructor client for LLM-powered relevance assessment
        threshold: Minimum confidence score for relevance (0-1). Defaults to 0.5
        model: LLM model to use for relevance assessment. Uses Config.get_fast_model() if not specified

    Returns:
        dict: Filtered results with same structure as input but only relevant documents:
            {
                "ids": [filtered_ids],
                "documents": [filtered_documents],
                "distances": [filtered_distances],
                "metadatas": [filtered_metadatas],
                "filtering_metadata": {
                    "original_count": int,
                    "filtered_count": int,
                    "threshold": float,
                    "assessments": [
                        {
                            "index": int,
                            "is_relevant": bool,
                            "confidence": float,
                            "reasoning": str
                        }
                    ]
                },
                # Any additional keys from original results are preserved
            }

    Raises:
        ValueError: If results structure is invalid or client is missing

    Example:
        results = retrieve_embeddings(collection, "parking rules", n_results=10)
        filtered = filter_results(client, results, "parking rules", threshold=0.7)
        print(f"Filtered from {filtered['filtering_metadata']['original_count']} "
              f"to {filtered['filtering_metadata']['filtered_count']} results")
    """
    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    if results is None:
        logger.error("Invalid results structure")
        raise ValueError("Invalid results structure")

    if client is None:
        logger.error("Client is required for result filtering")
        raise ValueError("Client is required for result filtering")

    required_keys = {"ids", "documents", "distances"}
    missing_keys = required_keys - set(results.keys())
    if missing_keys:
        logger.error(f"Results missing required keys: {missing_keys}")
        raise ValueError(f"Results missing required keys: {missing_keys}")

    logger.info(
        f"Filtering {len(results['ids'][0])} results for query: '{query[:30]}...'"
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [None])[0]

    original_count = len(ids)
    assessments = []

    # Assess relevance for each document
    for i, (doc_id, document, distance) in enumerate(zip(ids, documents, distances)):
        try:
            assessment = is_relevant(client, query, document, model)
            assessments.append(
                {
                    "index": i,
                    "is_relevant": assessment.is_relevant,
                    "confidence": assessment.confidence,
                    "reasoning": assessment.reasoning,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to assess document {i}: {str(e)}")
            # Mark as not relevant on failure
            assessments.append(
                {
                    "index": i,
                    "is_relevant": False,
                    "confidence": 0.0,
                    "reasoning": f"Assessment failed: {str(e)}",
                }
            )

    # Filter results based on relevance and threshold
    filtered_indices = []
    for i, assessment in enumerate(assessments):
        if assessment["is_relevant"] and assessment["confidence"] >= threshold:
            filtered_indices.append(i)

    filtered_ids = [ids[i] for i in filtered_indices]
    filtered_documents = [documents[i] for i in filtered_indices]
    filtered_distances = [distances[i] for i in filtered_indices]
    filtered_metadatas = [metadatas[i] if metadatas else None for i in filtered_indices]

    filtered_results = {
        "ids": [filtered_ids],
        "documents": [filtered_documents],
        "distances": [filtered_distances],
        "metadatas": [filtered_metadatas],
        "filtering_metadata": {
            "original_count": original_count,
            "filtered_count": len(filtered_indices),
            "threshold": threshold,
            "assessments": assessments,
        },
    }

    # Preserve any additional keys from original results
    for key, value in results.items():
        if key not in {"ids", "documents", "distances", "metadatas"}:
            filtered_results[key] = value

    filtered_count = len(filtered_indices)
    logger.info(
        f"Filtered {original_count} results to {filtered_count} relevant results "
        f"(threshold: {threshold})"
    )

    return filtered_results


def retrieve_embeddings(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 10,
    jurisdiction_id: str | None = None,
    where: dict | None = None,
    where_document: dict | None = None,
    rewrite: bool = False,
    rewrite_client: Instructor | None = None,
    rewrite_model: str | None = None,
    embedding_client: Any = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Retrieve similar documents from the embedding index using semantic search.

    Args:
        collection: ChromaDB collection to query
        query_text: Text to search for
        n_results: Number of results to return. Defaults to 10
        jurisdiction_id: Filter by specific jurisdiction (e.g., 'IL-WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        rewrite: Whether to apply HYDE query rewriting. Defaults to False
        rewrite_client: Instructor client for LLM-powered HYDE rewriting
        rewrite_model: LLM model to use for HYDE rewriting. Uses Config.get_fast_model() if not specified
        embedding_client: Embedding client for generating query embeddings. Defaults to None (uses configured provider)
        embedding_model: Embedding model name. Uses provider default if not specified

    Returns:
        dict: Query results containing documents, metadata, distances, and IDs

    Example:
        # Retrieve from specific jurisdiction
        results = retrieve_embeddings(collection, "parking regulations", jurisdiction_id="IL-WindyCity")

        # Retrieve with LLM-powered HYDE rewriting
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        results = retrieve_embeddings(
            collection,
            "where can I park my car",
            rewrite=True,
            client=client
        )

        # Retrieve from multiple jurisdictions
        results = retrieve_embeddings(
            collection,
            "zoning laws",
            where={"jurisdiction_id": {"$in": ["IL-WindyCity", "CA-LosAngeles"]}}
        )

        # Cross-jurisdiction comparison (no jurisdiction filter)
        results = retrieve_embeddings(collection, "noise ordinances", n_results=50)
    """
    # Use default model if not specified
    if rewrite_model is None:
        rewrite_model = Config.get_fast_model()

    # Apply HYDE rewriting if requested
    if rewrite:
        if rewrite_client is None:
            logger.error("Client is required for HYDE rewriting")
            raise ValueError("Client is required for HYDE rewriting")

        original_query = query_text
        result = hyde_rewriter(query_text, rewrite_client, rewrite_model)
        query_text = result.rewritten_query
        logger.debug(f"Applied HYDE rewrite: '{original_query}' -> '{query_text}'")

    logger.info(f"Retrieving embeddings for: '{query_text[:50]}...'")

    # Combine jurisdiction filter with additional where filters
    combined_where: dict[str, Any] | None = None
    if jurisdiction_id and where:
        # Both types of filters - combine with AND
        combined_where = {"$and": [{"jurisdiction_id": jurisdiction_id}, where]}
        logger.debug(f"Combined filters: {combined_where}")
    elif jurisdiction_id:
        combined_where = {"jurisdiction_id": jurisdiction_id}
        logger.debug(f"Using jurisdiction filter only: {jurisdiction_id}")
    elif where:
        combined_where = where
        logger.debug(f"Using custom filters only: {where}")

    # Generate embeddings explicitly to avoid dimension mismatch
    if embedding_client is None:
        # Use the proper embedding client factory function
        embedding_client = get_embedding_client()

    query_embeddings = get_embeddings(embedding_client, [query_text], embedding_model)
    # Cast to Any to satisfy ChromaDB typing expectations (avoids invariant list/ndarray mismatch)
    query_embeddings_any = cast(Any, query_embeddings)

    results = collection.query(
        query_embeddings=query_embeddings_any,
        n_results=n_results,
        where=combined_where,
        where_document=where_document,
    )

    result_count = len(results["ids"][0]) if results["ids"] else 0
    logger.info(f"Returned {result_count} results")

    # Log jurisdiction breakdown if possible
    if result_count > 0 and results.get("metadatas"):
        metadata_results = results["metadatas"]  # ChromaDB API returns 'metadatas'
        if metadata_results and metadata_results[0]:
            metadata_list = metadata_results[0]
            jurisdictions = set()
            states = set()
            municipalities = set()

            for metadata in metadata_list:
                if metadata:
                    if "jurisdiction_id" in metadata:
                        jurisdictions.add(metadata["jurisdiction_id"])
                    if "state" in metadata:
                        states.add(metadata["state"])
                    if "municipality" in metadata:
                        municipalities.add(metadata["municipality"])

            if jurisdictions:
                logger.debug(f"Results from jurisdictions: {sorted(jurisdictions)}")
            if states:
                logger.debug(f"Results from states: {sorted(states)}")
            if municipalities:
                logger.debug(f"Results from municipalities: {sorted(municipalities)}")

    return cast(dict[str, Any], results)


def get_jurisdiction_stats(collection: chromadb.Collection) -> dict:
    """Get statistics about embeddings per jurisdiction.

    Args:
        collection: ChromaDB collection to analyze

    Returns:
        dict: Statistics including counts per jurisdiction, state, and municipality
    """
    logger.info("Getting jurisdiction statistics from collection")

    try:
        # Get all documents to analyze metadata
        all_results = collection.get(include=["metadatas"])

        if not all_results or not all_results.get("metadatas"):
            logger.warning("No metadata found in collection")
            return {}

        metadata_list = all_results["metadatas"]  # ChromaDB API returns 'metadatas'
        if not metadata_list:
            return {}

        # Analyze jurisdiction distribution
        jurisdiction_counts = {}
        state_counts = {}
        municipality_counts = {}

        for metadata in metadata_list:
            if not metadata:
                continue

            if "jurisdiction_id" in metadata:
                jur_id = metadata["jurisdiction_id"]
                jurisdiction_counts[jur_id] = jurisdiction_counts.get(jur_id, 0) + 1

            if "state" in metadata:
                state = metadata["state"]
                state_counts[state] = state_counts.get(state, 0) + 1

            if "municipality" in metadata:
                municipality = metadata["municipality"]
                municipality_counts[municipality] = (
                    municipality_counts.get(municipality, 0) + 1
                )

        stats = {
            "total_documents": len(metadata_list),
            "jurisdictions": jurisdiction_counts,
            "states": state_counts,
            "municipalities": municipality_counts,
        }

        logger.info(f"Collection stats: {stats['total_documents']} total documents")
        logger.info(f"  Jurisdictions: {len(jurisdiction_counts)}")
        logger.info(f"  States: {len(state_counts)}")
        logger.info(f"  Municipalities: {len(municipality_counts)}")

        return stats

    except Exception as e:
        logger.error(f"Failed to get jurisdiction stats: {str(e)}")
        return {}


def compare_jurisdictions(
    collection: chromadb.Collection,
    query: str,
    jurisdictions: list[str],
    n_per_jurisdiction: int = 5,
) -> dict:
    """Compare how different jurisdictions handle the same legal topic.

    Args:
        collection: ChromaDB collection to query
        query: Legal topic to compare across jurisdictions
        jurisdictions: List of jurisdiction IDs to compare
        n_per_jurisdiction: Number of results per jurisdiction

    Returns:
        dict: Results organized by jurisdiction
    """
    logger.info(f"Comparing jurisdictions for query: '{query}'")
    logger.info(f"Jurisdictions: {jurisdictions}")

    comparison_results = {}

    for jurisdiction_id in jurisdictions:
        logger.debug(f"Querying jurisdiction: {jurisdiction_id}")

        results = retrieve_embeddings(
            collection=collection,
            query_text=query,
            jurisdiction_id=jurisdiction_id,
            n_results=n_per_jurisdiction,
        )

        comparison_results[jurisdiction_id] = results

        result_count = len(results["ids"][0]) if results["ids"] else 0
        logger.info(f"  {jurisdiction_id}: {result_count} results")

    logger.info(f"Comparison completed for {len(jurisdictions)} jurisdictions")
    return comparison_results


def retrieve_sections(
    collection: chromadb.Collection,
    query_text: str,
    sections_parquet_path: str | Path,
    n_results: int = 10,
    jurisdiction_id: str | None = None,
    where: dict | None = None,
    where_document: dict | None = None,
    rewrite: bool = False,
    rewrite_client: Instructor | None = None,
    rewrite_model: str | None = None,
    embedding_client=None,
    embedding_model: str = "embeddinggemma",
) -> dict:
    """Retrieve sections by searching embeddings at segment level but returning full section context.

    This function performs semantic search at the segment level for precision, then aggregates
    the results by their parent sections to provide broader legal context. Each result includes
    the full section content along with the specific matching segments.

    Args:
        collection: ChromaDB collection to query
        query_text: Text to search for
        sections_parquet_path: Path to sections.parquet file containing section data
        n_results: Number of segment results to retrieve. Defaults to 10
        jurisdiction_id: Filter by specific jurisdiction (e.g., 'IL-WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        rewrite: Whether to apply HYDE query rewriting. Defaults to False
        rewrite_client: Instructor client for LLM-powered HYDE rewriting
        rewrite_model: LLM model to use for HYDE rewriting. Uses Config.get_fast_model() if not specified
        embedding_client: Embedding client for generating query embeddings. Defaults to None (uses configured provider)
        embedding_model: Embedding model name. Uses provider default if not specified

    Returns:
        dict: Section-level results with structure:
            {
                "sections": [
                    {
                        "section_idx": int,
                        "heading_text": str,
                        "body_text": str,
                        "heading_level": int,
                        "parent": Optional[int],
                        "matching_segments": [
                            {
                                "segment_idx": int,
                                "segment_text": str,
                                "distance": float,
                                "segment_position": int
                            }
                        ],
                        "relevance_score": float,  # Best segment score
                        "segment_count": int
                    }
                ],
                "query_info": {
                    "original_query": str,
                    "rewritten_query": Optional[str],
                    "total_segments_found": int,
                    "unique_sections": int
                }
            }

    Raises:
        ValueError: If sections_parquet_path doesn't exist or required columns are missing
        FileNotFoundError: If sections parquet file cannot be found

    Example:
        # Basic section retrieval
        results = retrieve_sections(
            collection,
            "parking regulations",
            sections_parquet_path="data/laws/IL-WindyCity/tables/sections.parquet"
        )

        # Section retrieval with HYDE rewriting
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        results = retrieve_sections(
            collection,
            "where can I park my car",
            sections_parquet_path="data/laws/IL-WindyCity/tables/sections.parquet",
            rewrite=True,
            client=client
        )

        # Access section content and matching segments
        for section in results["sections"]:
            print(f"Section: {section['heading_text']}")
            print(f"Content: {section['body_text'][:100]}...")
            print(f"Found {section['segment_count']} matching segments")
            for segment in section["matching_segments"]:
                print(f"  Segment: {segment['segment_text'][:50]}...")
    """
    # Use default model if not specified
    if rewrite_model is None:
        rewrite_model = Config.get_fast_model()

    logger.info(f"Retrieving sections for query: '{query_text[:50]}...'")

    sections_path = Path(sections_parquet_path)

    if not sections_path.exists():
        logger.error(f"Sections parquet file not found: {sections_path}")
        raise FileNotFoundError(f"Sections parquet file not found: {sections_path}")

    logger.debug("Step 1: Retrieving segment-level results")
    segment_results = retrieve_embeddings(
        collection=collection,
        query_text=query_text,
        n_results=n_results,
        jurisdiction_id=jurisdiction_id,
        where=where,
        where_document=where_document,
        rewrite=rewrite,
        rewrite_client=rewrite_client,
        rewrite_model=rewrite_model,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )

    original_query = query_text
    rewritten_query = segment_results.get("rewritten_query") if rewrite else None

    # Check if we have any results
    if not segment_results.get("ids") or not segment_results["ids"][0]:
        logger.info("No segment results found")
        return {
            "sections": [],
            "query_info": {
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "total_segments_found": 0,
                "unique_sections": 0,
            },
        }

    total_segments_found = len(segment_results["ids"][0])
    logger.info(f"Found {total_segments_found} segment results")

    logger.debug("Step 2: Processing segment results")

    segment_ids = segment_results["ids"][0]
    segment_documents = segment_results["documents"][0]
    segment_distances = segment_results["distances"][0]
    segment_metadatas = segment_results.get("metadatas", [None])[0]

    # Group segments by section_ref
    sections_to_segments: dict[int, list[dict[str, Any]]] = {}

    for i, seg_id in enumerate(segment_ids):
        metadata = (
            segment_metadatas[i]
            if segment_metadatas and i < len(segment_metadatas)
            else {}
        )

        section_ref = metadata.get("section_ref")
        if section_ref is None:
            logger.warning(f"Segment {seg_id} missing section_ref in metadata")
            continue

        segment_data = {
            "segment_idx": int(seg_id),
            "segment_text": segment_documents[i],
            "distance": segment_distances[i],
            "segment_position": metadata.get("segment_position", 0),
            "section_heading": metadata.get("section_heading", ""),
            "section_level": metadata.get("section_level", 1),
        }

        # Group by section
        if section_ref not in sections_to_segments:
            sections_to_segments[section_ref] = []
        sections_to_segments[section_ref].append(segment_data)

    unique_sections = len(sections_to_segments)
    logger.info(f"Grouped segments into {unique_sections} unique sections")

    if not sections_to_segments:
        logger.warning("No valid section references found in segment metadata")
        return {
            "sections": [],
            "query_info": {
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "total_segments_found": total_segments_found,
                "unique_sections": 0,
            },
        }

    logger.debug("Step 3: Loading sections data from parquet")

    try:
        # Load sections DataFrame
        sections_df = pl.read_parquet(sections_path)
        logger.debug(f"Loaded {len(sections_df)} sections from parquet")

        # Validate required columns exist
        required_columns = {"section_idx", "heading_text", "body_text", "heading_level"}
        missing_columns = required_columns - set(sections_df.columns)
        if missing_columns:
            logger.error(
                f"Sections parquet missing required columns: {missing_columns}"
            )
            raise ValueError(
                f"Sections parquet missing required columns: {missing_columns}"
            )

        # Filter to only sections we have results for
        section_indices = list(sections_to_segments.keys())
        filtered_sections_df = sections_df.filter(
            pl.col("section_idx").is_in(section_indices)
        )

        logger.debug(f"Filtered to {len(filtered_sections_df)} matching sections")

        # Convert to dictionary for easier lookup
        sections_dict = {}
        for row in filtered_sections_df.to_dicts():
            sections_dict[row["section_idx"]] = row

    except Exception as e:
        logger.error(f"Failed to load sections parquet: {str(e)}")
        raise ValueError(f"Failed to load sections parquet: {str(e)}") from e

    logger.debug("Step 4: Building section-level results")

    section_results = []

    for section_idx, segments in sections_to_segments.items():
        # Get section data
        section_data = sections_dict.get(section_idx)
        if not section_data:
            logger.warning(f"Section {section_idx} not found in parquet data")
            continue

        # Calculate relevance score (best segment distance)
        best_distance = min(seg["distance"] for seg in segments)

        # Sort segments by distance (most relevant first)
        segments_sorted = sorted(segments, key=lambda x: x["distance"])

        section_result = {
            "section_idx": section_idx,
            "heading_text": section_data["heading_text"],
            "body_text": section_data["body_text"],
            "heading_level": section_data["heading_level"],
            "parent": section_data.get("parent"),
            "matching_segments": [
                {
                    "segment_idx": seg["segment_idx"],
                    "segment_text": seg["segment_text"],
                    "distance": seg["distance"],
                    "segment_position": seg["segment_position"],
                }
                for seg in segments_sorted
            ],
            "relevance_score": best_distance,
            "segment_count": len(segments),
        }

        section_results.append(section_result)

    # Sort sections by relevance score (best first)
    section_results.sort(key=lambda x: x["relevance_score"])

    logger.info(f"Returning {len(section_results)} sections with context")

    return {
        "sections": section_results,
        "query_info": {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "total_segments_found": total_segments_found,
            "unique_sections": len(section_results),
        },
    }
