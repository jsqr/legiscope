import chromadb
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from instructor import Instructor
from loguru import logger
import polars as pl

from legiscope.utils import ask
from legiscope.embeddings import get_embeddings, EmbeddingClient


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
        description="Type of legal query (e.g., 'parking', 'zoning', 'permit', 'licensing')"
    )


def hyde_rewriter(
    query: str, client: Instructor, model: str = "gpt-4.1-mini"
) -> HydeRewrite:
    """Rewrite a natural language query into municipal code style text using HYDE approach.

    Transforms user queries into the style and format of municipal code text to improve
    semantic similarity matching with embedded legal documents using LLM-powered transformation.

    Args:
        query: Natural language query from user
        client: Instructor client for LLM-powered rewriting
        model: LLM model to use. Defaults to 'gpt-4.1-mini'

    Returns:
        HydeRewrite: Structured response with rewritten query and metadata

    Raises:
        ValueError: If query is empty or client is invalid

    Example:
        import instructor
        from openai import OpenAI
        client = instructor.from_openai(OpenAI())
        result = hyde_rewriter("where can I park my car", client)
        print(result.rewritten_query)
        print(result.confidence)
        print(result.query_type)
    """
    # Validate input
    if not query or not query.strip():
        logger.error("Query cannot be empty for HYDE rewriting")
        raise ValueError("Query cannot be empty for HYDE rewriting")

    if client is None:
        logger.error("Client is required for HYDE rewriting")
        raise ValueError("Client is required for HYDE rewriting")

    logger.info(f"Using LLM for HYDE rewrite: '{query[:50]}...'")

    # System prompt for HYDE query rewriting
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

Classify the query type (e.g., parking, zoning, permits, licensing, noise, animals, etc.)
and provide a confidence score for the rewrite quality."""

    # User prompt with the query
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


def retrieve_embeddings(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 10,
    jurisdiction_id: str | None = None,
    state: str | None = None,
    municipality: str | None = None,
    where: dict | None = None,
    where_document: dict | None = None,
    rewrite: bool = False,
    client: Instructor | None = None,
    model: str = "gpt-4.1-mini",
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = "embeddinggemma",
) -> Dict[str, Any]:
    """Retrieve similar documents from the embedding index using semantic search.

    Args:
        collection: ChromaDB collection to query
        query_text: Text to search for
        n_results: Number of results to return. Defaults to 10
        jurisdiction_id: Filter by specific jurisdiction (e.g., 'IL-WindyCity')
        state: Filter by state only (e.g., 'IL')
        municipality: Filter by municipality only (e.g., 'WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        rewrite: Whether to apply HYDE query rewriting. Defaults to False
        client: Instructor client for LLM-powered HYDE rewriting
        model: LLM model to use for HYDE rewriting. Defaults to 'gpt-4.1-mini'
        embedding_client: Embedding client for generating query embeddings. Defaults to None (uses ollama)
        embedding_model: Embedding model name. Defaults to 'embeddinggemma'

    Returns:
        dict: Query results containing documents, metadata, distances, and IDs

    Example:
        # Retrieve from specific jurisdiction
        results = retrieve_embeddings(collection, "parking regulations", jurisdiction_id="IL-WindyCity")

        # Retrieve with LLM-powered HYDE rewriting
        import instructor
        from openai import OpenAI
        client = instructor.from_openai(OpenAI())
        results = retrieve_embeddings(
            collection,
            "where can I park my car",
            rewrite=True,
            client=client,
            model="gpt-4.1-mini"
        )

        # Retrieve from all Illinois municipalities
        results = retrieve_embeddings(collection, "business licenses", state="IL")

        # Retrieve from multiple jurisdictions
        results = retrieve_embeddings(
            collection,
            "zoning laws",
            where={"jurisdiction_id": {"$in": ["IL-WindyCity", "CA-LosAngeles"]}}
        )

        # Cross-jurisdiction comparison (no jurisdiction filter)
        results = retrieve_embeddings(collection, "noise ordinances", n_results=50)
    """
    # Apply HYDE rewriting if requested
    if rewrite:
        if client is None:
            logger.error("Client is required for HYDE rewriting")
            raise ValueError("Client is required for HYDE rewriting")

        original_query = query_text
        result = hyde_rewriter(query_text, client, model)
        query_text = result.rewritten_query
        logger.debug(f"Applied HYDE rewrite: '{original_query}' -> '{query_text}'")

    logger.info(f"Retrieving embeddings for: '{query_text[:50]}...'")

    # Build jurisdiction filters
    jurisdiction_filters = {}

    if jurisdiction_id:
        jurisdiction_filters["jurisdiction_id"] = jurisdiction_id
        logger.debug(f"Filtering by jurisdiction_id: {jurisdiction_id}")
    elif state or municipality:
        if state:
            jurisdiction_filters["state"] = state
            logger.debug(f"Filtering by state: {state}")
        if municipality:
            jurisdiction_filters["municipality"] = municipality
            logger.debug(f"Filtering by municipality: {municipality}")

    # Combine jurisdiction filters with additional where filters
    combined_where: Dict[str, Any] | None = None
    if jurisdiction_filters and where:
        # Both types of filters - combine with AND
        combined_where = {"$and": [jurisdiction_filters, where]}
        logger.debug(f"Combined filters: {combined_where}")
    elif jurisdiction_filters:
        combined_where = jurisdiction_filters
        logger.debug(f"Using jurisdiction filters only: {jurisdiction_filters}")
    elif where:
        combined_where = where
        logger.debug(f"Using custom filters only: {where}")

    # Generate embeddings explicitly to avoid dimension mismatch
    if embedding_client is None:
        # Try to import ollama as default embedding client
        try:
            import ollama

            embedding_client = ollama.Client()  # type: ignore
        except ImportError:
            logger.error("No embedding client provided and ollama not available")
            raise ValueError("Embedding client is required for querying")

    # Generate embedding for the query
    # Type ignore because ollama.Client is compatible with EmbeddingClient protocol
    query_embeddings = get_embeddings(embedding_client, [query_text], embedding_model)  # type: ignore[arg-type]

    results = collection.query(
        query_embeddings=query_embeddings,  # type: ignore
        n_results=n_results,
        where=combined_where,  # type: ignore
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

    return results  # type: ignore


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
    state: str | None = None,
    municipality: str | None = None,
    where: dict | None = None,
    where_document: dict | None = None,
    rewrite: bool = False,
    client: Instructor | None = None,
    model: str = "gpt-4.1-mini",
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
        state: Filter by state only (e.g., 'IL')
        municipality: Filter by municipality only (e.g., 'WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        rewrite: Whether to apply HYDE query rewriting. Defaults to False
        client: Instructor client for LLM-powered HYDE rewriting
        model: LLM model to use for HYDE rewriting. Defaults to 'gpt-4.1-mini'

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
        import instructor
        from openai import OpenAI
        client = instructor.from_openai(OpenAI())
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
    logger.info(f"Retrieving sections for query: '{query_text[:50]}...'")

    # Convert path to Path object for consistency
    sections_path = Path(sections_parquet_path)

    # Validate sections parquet file exists
    if not sections_path.exists():
        logger.error(f"Sections parquet file not found: {sections_path}")
        raise FileNotFoundError(f"Sections parquet file not found: {sections_path}")

    # Step 1: Retrieve segment-level results using existing function
    logger.debug("Step 1: Retrieving segment-level results")
    segment_results = retrieve_embeddings(
        collection=collection,
        query_text=query_text,
        n_results=n_results,
        jurisdiction_id=jurisdiction_id,
        state=state,
        municipality=municipality,
        where=where,
        where_document=where_document,
        rewrite=rewrite,
        client=client,
        model=model,
    )

    # Extract query information
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

    # Step 2: Extract segment data and group by section
    logger.debug("Step 2: Processing segment results")

    segment_ids = segment_results["ids"][0]
    segment_documents = segment_results["documents"][0]
    segment_distances = segment_results["distances"][0]
    segment_metadatas = segment_results.get("metadatas", [None])[0]

    # Group segments by section_ref
    sections_to_segments: Dict[int, List[Dict[str, Any]]] = {}

    for i, seg_id in enumerate(segment_ids):
        # Get metadata for this segment
        metadata = (
            segment_metadatas[i]
            if segment_metadatas and i < len(segment_metadatas)
            else {}
        )

        # Extract section_ref from metadata
        section_ref = metadata.get("section_ref")
        if section_ref is None:
            logger.warning(f"Segment {seg_id} missing section_ref in metadata")
            continue

        # Create segment data
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

    # Step 3: Load sections data from parquet
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

    # Step 4: Build final results
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

        # Build section result
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

    # Step 5: Return final results
    return {
        "sections": section_results,
        "query_info": {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "total_segments_found": total_segments_found,
            "unique_sections": len(section_results),
        },
    }
