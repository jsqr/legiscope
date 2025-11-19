"""
Query processing module for the legiscope package.
"""

from typing import Any

import polars as pl
from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.llm_config import Config
from legiscope.retrieve import retrieve_sections, filter_sections
from legiscope.utils import ask


class LegalQueryResponse(BaseModel):
    """Structured response for legal queries with citations and reasoning."""

    short_answer: str = Field(
        description="A concise, direct answer to the user's legal question"
    )
    reasoning: str = Field(
        description="Detailed explanation of the legal reasoning used to arrive at the answer"
    )
    citations: list[str] = Field(
        description="List of specific legal sections or provisions that support the answer"
    )
    supporting_passages: list[str] = Field(
        description="Direct excerpts from the retrieved legal text that support the reasoning"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the answer based on the available evidence",
        ge=0.0,
        le=1.0,
    )
    limitations: str = Field(
        description="Any limitations or caveats to the answer based on the available information"
    )


def query_legal_documents(
    client: Instructor,
    query: str,
    retrieval_results: dict[str, Any],
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    filter_relevance: bool = False,
    relevance_threshold: float = 0.5,
    filter_model: str | None = None,
) -> LegalQueryResponse:
    """
    Process a user query against retrieved legal documents using LLM analysis.

    Takes the filtered results from a retrieval operation and generates a comprehensive
    response with legal reasoning, citations, and supporting evidence.

    Args:
        client: Instructor client for LLM-powered analysis
        query: The user's legal question or query
        retrieval_results: Results from retrieve_sections or similar retrieval functions
        model: LLM model to use. Uses Config.get_fast_model() if not specified
        temperature: Sampling temperature for the LLM. Defaults to 0.1
        max_retries: Maximum retry attempts for LLM calls. Defaults to 3
        filter_relevance: Whether to filter sections by relevance before LLM processing. Defaults to False
        relevance_threshold: Minimum confidence score for relevance filtering (0-1). Defaults to 0.5
        filter_model: LLM model to use for relevance filtering. Uses Config.get_fast_model() if not specified

    Returns:
        LegalQueryResponse: Structured response with answer, reasoning, citations, and evidence

    Raises:
        ValueError: If client is invalid, query is empty, or results structure is invalid
        instructor.exceptions.InstructorError: If LLM call fails

    Example:
        from legiscope.llm_config import Config
        from legiscope.retrieve import retrieve_sections
        from legiscope.query import query_legal_documents

        # Setup client
        client = Config.get_fast_client()

        # Retrieve relevant sections
        results = retrieve_sections(
            collection=collection,
            query_text="Are there restrictions on drug paraphernalia sales?",
            sections_parquet_path="./data/laws/IL-WindyCity/tables/sections.parquet",
            jurisdiction_id="IL-WindyCity"
        )

        # Process query with relevance filtering
        response = query_legal_documents(
            client=client,
            query="Are there restrictions on drug paraphernalia sales?",
            retrieval_results=results,
            filter_relevance=True,
            relevance_threshold=0.7
        )

        print(f"Answer: {response.short_answer}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Citations: {response.citations}")
    """
    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    # 1. Input validation
    _validate_query_inputs(
        client=client,
        query=query,
        retrieval_results=retrieval_results,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        filter_relevance=filter_relevance,
        relevance_threshold=relevance_threshold,
        filter_model=filter_model,
    )

    logger.info(f"Processing query: '{query[:50]}...'")
    logger.debug(f"Using model: {model}, temperature: {temperature}")

    # 2. Extract and validate sections
    sections = _extract_and_validate_sections(retrieval_results)
    if not sections:
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found.",
            reasoning="The search did not return any legal sections that address your query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available to answer query.",
        )

    # 3. Apply relevance filtering
    sections = _apply_relevance_filtering(
        retrieval_results=retrieval_results,
        query=query,
        client=client,
        filter_relevance=filter_relevance,
        relevance_threshold=relevance_threshold,
        filter_model=filter_model,
    )

    if not sections:
        logger.warning("All sections filtered out as irrelevant")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
            reasoning="The search returned legal sections, but all were determined to be irrelevant to your specific query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available after relevance filtering.",
        )

    # 4. Prepare context
    full_context = _prepare_legal_context(sections)

    # 5. Build prompts
    system_prompt, user_prompt = _build_legal_prompts(query, full_context)

    # 6. Execute LLM call
    return _execute_query_llm_call(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def format_query_response(response: LegalQueryResponse) -> str:
    """
    Format a LegalQueryResponse for display.

    Args:
        response: The LegalQueryResponse to format

    Returns:
        str: Formatted response string
    """
    formatted = f"""
## Legal Analysis

**Answer:** {response.short_answer}

**Confidence:** {response.confidence:.1%}

### Reasoning
{response.reasoning}

### Citations
"""
    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            formatted += f"{i}. {citation}\n"
    else:
        formatted += "No specific citations available.\n"

    formatted += "\n### Supporting Passages\n"
    if response.supporting_passages:
        for i, passage in enumerate(response.supporting_passages, 1):
            formatted += f'{i}. "{passage}"\n'
    else:
        formatted += "No supporting passages available.\n"

    if response.limitations:
        formatted += f"\n### Limitations\n{response.limitations}\n"

    return formatted.strip()


def run_queries(
    client: Instructor,
    queries: list[str],
    jurisdiction_id: str,
    sections_parquet_path: str,
    collection,
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    n_results: int = 10,
    use_hyde: bool = False,
    filter_relevance: bool = False,
    relevance_threshold: float = 0.5,
    filter_model: str | None = None,
) -> pl.DataFrame:
    """
    Run multiple queries against a jurisdiction and compile results in a structured DataFrame.

    Processes a list of queries by retrieving relevant sections for each query and
    generating structured legal responses. Results are compiled into a DataFrame for
    easy analysis and comparison.

    Args:
        client: Instructor client for LLM-powered analysis
        queries: List of legal questions to process
        jurisdiction_id: Jurisdiction identifier (e.g., 'IL-WindyCity')
        sections_parquet_path: Path to sections.parquet file containing section data
        collection: ChromaDB collection to query
        model: LLM model to use for query processing. Uses Config.get_fast_model() if not specified
        temperature: Sampling temperature for LLM. Defaults to 0.1
        max_retries: Maximum retry attempts for LLM calls. Defaults to 3
        n_results: Number of results to retrieve per query. Defaults to 10
        use_hyde: Whether to apply HYDE query rewriting. Defaults to False
        filter_relevance: Whether to filter sections by relevance before LLM processing. Defaults to False
        relevance_threshold: Minimum confidence score for relevance filtering (0-1). Defaults to 0.5
        filter_model: LLM model to use for relevance filtering. Uses Config.get_fast_model() if not specified

    Returns:
        pl.DataFrame: Structured results with columns:
            - query: Original query string
            - short_answer: Concise answer to the query
            - reasoning: Detailed legal reasoning
            - citations: List of legal citations (as string)
            - supporting_passages: List of supporting passages (as string)
            - confidence: Confidence score (0-1)
            - limitations: Any limitations or caveats
            - sections_found: Number of relevant sections found
            - segments_found: Number of matching segments found
            - processing_time: Time taken to process query (in seconds)

    Raises:
        ValueError: If required parameters are missing or invalid
        instructor.exceptions.InstructorError: If LLM calls fail

    Example:
        from legiscope.llm_config import Config
        from legiscope.query import run_queries
        import chromadb

        # Setup
        client = Config.get_fast_client()
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = chroma_client.get_collection("legal_code_all")

        # Run multiple queries with relevance filtering
        queries = [
            "Are there restrictions on drug paraphernalia sales?",
            "What are the parking regulations?",
            "Do I need a permit for home business?"
        ]

        results_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="./data/laws/IL-WindyCity/tables/sections.parquet",
            collection=collection,
            model=Config.get_powerful_model(),
            filter_relevance=True,
            relevance_threshold=0.7
        )

        # View results
        print(results_df.select(["query", "short_answer", "confidence"]))
    """
    import time

    # 1. Input validation
    _validate_batch_query_inputs(
        client=client,
        queries=queries,
        jurisdiction_id=jurisdiction_id,
        sections_parquet_path=sections_parquet_path,
        collection=collection,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        n_results=n_results,
        use_hyde=use_hyde,
        filter_relevance=filter_relevance,
        relevance_threshold=relevance_threshold,
        filter_model=filter_model,
    )

    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    logger.info(
        f"Processing {len(queries)} queries for jurisdiction: {jurisdiction_id}"
    )
    logger.debug(f"Using model: {model}, n_results: {n_results}, use_hyde: {use_hyde}")

    # 2. Process queries in loop
    results = []
    for i, query in enumerate(queries):
        if query is None or not isinstance(query, str) or not query.strip():
            logger.warning(f"Skipping empty query at index {i}")
            continue

        start_time = time.time()
        logger.info(f"Processing query {i + 1}/{len(queries)}: '{query[:50]}...'")

        result = _process_single_query_with_error_handling(
            client=client,
            query=query,
            jurisdiction_id=jurisdiction_id,
            sections_parquet_path=sections_parquet_path,
            collection=collection,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            n_results=n_results,
            use_hyde=use_hyde,
            filter_relevance=filter_relevance,
            relevance_threshold=relevance_threshold,
            filter_model=filter_model,
            start_time=start_time,
        )

        results.append(result)

        if "Error:" not in result["short_answer"]:
            logger.info(
                f"Query {i + 1} completed - confidence: {result['confidence']:.2f}, "
                f"sections: {result['sections_found']}, time: {result['processing_time']:.2f}s"
            )

    # 3. Compile and return results
    return _compile_query_results(results)


def _validate_query_inputs(
    client: Instructor,
    query: str,
    retrieval_results: dict[str, Any],
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    filter_relevance: bool = False,
    relevance_threshold: float = 0.5,
    filter_model: str | None = None,
) -> None:
    """Validate inputs for query_legal_documents function."""
    if not client:
        logger.error("Client is required for query processing")
        raise ValueError("Client is required for query processing")

    if not query or not query.strip():
        logger.error("Query cannot be empty for query processing")
        raise ValueError("Query cannot be empty for query processing")

    if not retrieval_results:
        logger.error("Retrieval results are required for query processing")
        raise ValueError("Retrieval results are required for query processing")

    if temperature < 0 or temperature > 2:
        logger.error("Temperature must be between 0 and 2")
        raise ValueError("Temperature must be between 0 and 2")

    if max_retries < 0:
        logger.error("Max retries must be non-negative")
        raise ValueError("Max retries must be non-negative")

    if relevance_threshold < 0 or relevance_threshold > 1:
        logger.error("Relevance threshold must be between 0 and 1")
        raise ValueError("Relevance threshold must be between 0 and 1")


def _extract_and_validate_sections(
    retrieval_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract and validate sections from retrieval results."""
    sections = retrieval_results.get("sections", [])
    if not sections:
        logger.warning("No sections found in retrieval results")
        return []

    logger.info(f"Found {len(sections)} relevant sections to analyze")
    return sections


def _apply_relevance_filtering(
    retrieval_results: dict[str, Any],
    query: str,
    client: Instructor,
    filter_relevance: bool = False,
    relevance_threshold: float = 0.5,
    filter_model: str | None = None,
) -> list[dict[str, Any]]:
    """Apply relevance filtering to sections if requested."""
    if not filter_relevance:
        return retrieval_results.get("sections", [])

    logger.info(f"Applying relevance filtering with threshold: {relevance_threshold}")

    # Use default model for filtering if not specified
    if filter_model is None:
        filter_model = Config.get_fast_model()

    try:
        filtered_results = filter_sections(
            client=client,
            sections_results=retrieval_results,
            query=query,
            confidence_threshold=relevance_threshold,
            model=filter_model,
        )

        sections = filtered_results.get("sections", [])
        original_count = filtered_results.get("original_count", 0)
        filtered_count = filtered_results.get("filtered_count", 0)

        reduction_percentage = (
            ((original_count - filtered_count) / original_count * 100)
            if original_count > 0
            else 0
        )

        logger.info(
            f"Relevance filtering complete: {original_count} -> {filtered_count} sections "
            f"({reduction_percentage:.1f}% reduction)"
        )

        return sections

    except Exception as e:
        logger.error(
            f"Relevance filtering failed, proceeding with original sections: {str(e)}"
        )
        # Continue with original sections if filtering fails
        return retrieval_results.get("sections", [])


def _prepare_legal_context(sections: list[dict[str, Any]]) -> str:
    """Prepare formatted context from sections for LLM processing."""
    context_sections = []
    for i, section in enumerate(sections):
        section_text = f"""
Section {i + 1}: {section.get("heading_text", "Untitled Section")}
Relevance Score: {section.get("relevance_score", 0):.3f}
Content: {section.get("body_text", "")}

Matching Segments:
"""
        # Add matching segments for context
        for j, segment in enumerate(section.get("matching_segments", [])):
            segment_text = segment.get("segment_text", "")
            if segment_text:
                section_text += f"  - Segment {j + 1}: {segment_text}\n"

        context_sections.append(section_text)

    return "\n".join(context_sections)


def _build_legal_prompts(query: str, full_context: str) -> tuple[str, str]:
    """Build system and user prompts for legal query processing."""
    system_prompt = """You are a lawyer specializing in municipal law and regulations.
Your task is to analyze the provided legal context and answer the user's question accurately.

Guidelines for your analysis:
1. Provide a direct, concise answer to the user's question
2. Explain your legal reasoning clearly and thoroughly
3. Cite specific sections or provisions that support your answer
4. Include direct excerpts from the legal text that support your reasoning
5. Assess your confidence based on the available evidence
6. Note any limitations or gaps in the available information

When citing sections, use the section headings provided in the context. When including
supporting passages, use direct quotes from the legal text that most strongly support
your reasoning.

Be precise and objective in your analysis. If the provided context does not contain
sufficient information to answer the question definitively, acknowledge this limitation
and provide the best answer possible with the available information."""

    user_prompt = f"""Please answer the following legal question based on the provided municipal code context:

User Question: "{query}"

Legal Context:
{full_context}

Please analyze this legal context and provide a comprehensive response following the guidelines."""

    return system_prompt, user_prompt


def _execute_query_llm_call(
    client: Instructor,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> LegalQueryResponse:
    """Execute LLM call for query processing."""
    try:
        logger.debug("Making LLM call for query processing")

        response = ask(
            client=client,
            prompt=user_prompt,
            response_model=LegalQueryResponse,
            system=system_prompt,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )

        logger.info(
            f"Query processing completed - confidence: {response.confidence:.2f}, "
            f"citations: {len(response.citations)}, supporting passages: {len(response.supporting_passages)}"
        )

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise


def _validate_batch_query_inputs(
    client: Instructor,
    queries: list[str],
    jurisdiction_id: str,
    sections_parquet_path: str,
    collection,
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    n_results: int = 10,
    use_hyde: bool = False,
    filter_relevance: bool = False,
    relevance_threshold: float = 0.5,
    filter_model: str | None = None,
) -> None:
    """Validate inputs for run_queries function."""
    if not client:
        logger.error("Client is required for query processing")
        raise ValueError("Client is required for query processing")

    if not queries or not isinstance(queries, list):
        logger.error("Queries must be a non-empty list")
        raise ValueError("Queries must be a non-empty list")

    if not jurisdiction_id or not jurisdiction_id.strip():
        logger.error("Jurisdiction ID is required")
        raise ValueError("Jurisdiction ID is required")

    if not sections_parquet_path:
        logger.error("Sections parquet path is required")
        raise ValueError("Sections parquet path is required")

    if collection is None:
        logger.error("ChromaDB collection is required")
        raise ValueError("ChromaDB collection is required")

    if n_results <= 0:
        logger.error("n_results must be positive")
        raise ValueError("n_results must be positive")


def _process_single_query_with_error_handling(
    client: Instructor,
    query: str,
    jurisdiction_id: str,
    sections_parquet_path: str,
    collection,
    model: str,
    temperature: float,
    max_retries: int,
    n_results: int,
    use_hyde: bool,
    filter_relevance: bool,
    relevance_threshold: float,
    filter_model: str | None,
    start_time: float,
) -> dict:
    """Process a single query with comprehensive error handling."""
    import time

    try:
        retrieval_results = retrieve_sections(
            collection=collection,
            query_text=query,
            sections_parquet_path=sections_parquet_path,
            n_results=n_results,
            jurisdiction_id=jurisdiction_id,
            rewrite=use_hyde,
            rewrite_client=client if use_hyde else None,
            rewrite_model=model,
        )

        query_info = retrieval_results.get("query_info", {})
        sections_found = len(retrieval_results.get("sections", []))
        segments_found = query_info.get("total_segments_found", 0)

        query_response = query_legal_documents(
            client=client,
            query=query,
            retrieval_results=retrieval_results,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            filter_relevance=filter_relevance,
            relevance_threshold=relevance_threshold,
            filter_model=filter_model,
        )

        processing_time = time.time() - start_time

        return {
            "query": query,
            "short_answer": query_response.short_answer,
            "reasoning": query_response.reasoning,
            "citations": str(
                query_response.citations
            ),  # Convert list to string for DataFrame
            "supporting_passages": str(query_response.supporting_passages),
            "confidence": query_response.confidence,
            "limitations": query_response.limitations,
            "sections_found": sections_found,
            "segments_found": segments_found,
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Query processing failed: {str(e)}")

        # Add failed result with error information
        return {
            "query": query,
            "short_answer": f"Error: {str(e)}",
            "reasoning": f"Query processing failed with error: {str(e)}",
            "citations": "[]",
            "supporting_passages": "[]",
            "confidence": 0.0,
            "limitations": f"Processing failed due to error: {str(e)}",
            "sections_found": 0,
            "segments_found": 0,
            "processing_time": processing_time,
        }


def _compile_query_results(results: list[dict]) -> pl.DataFrame:
    """Compile query results into a structured DataFrame."""
    if not results:
        logger.warning("No queries were processed successfully")
        return pl.DataFrame(
            schema={
                "query": pl.Utf8,
                "short_answer": pl.Utf8,
                "reasoning": pl.Utf8,
                "citations": pl.Utf8,
                "supporting_passages": pl.Utf8,
                "confidence": pl.Float64,
                "limitations": pl.Utf8,
                "sections_found": pl.Int64,
                "segments_found": pl.Int64,
                "processing_time": pl.Float64,
            }
        )

    df = pl.DataFrame(results)

    logger.info(f"Completed processing {len(results)} queries")
    logger.info(f"Average confidence: {df['confidence'].mean():.2f}")
    logger.info(f"Average processing time: {df['processing_time'].mean():.2f}s")

    return df
