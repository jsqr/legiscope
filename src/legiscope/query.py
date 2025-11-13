"""
Query processing module for the legiscope package.
"""

from typing import Any, Dict, List

import polars as pl
from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.retrieve import retrieve_sections
from legiscope.utils import ask
from legiscope.llm_config import Config


class LegalQueryResponse(BaseModel):
    """Structured response for legal queries with citations and reasoning."""

    short_answer: str = Field(
        description="A concise, direct answer to the user's legal question"
    )
    reasoning: str = Field(
        description="Detailed explanation of the legal reasoning used to arrive at the answer"
    )
    citations: List[str] = Field(
        description="List of specific legal sections or provisions that support the answer"
    )
    supporting_passages: List[str] = Field(
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
    retrieval_results: Dict[str, Any],
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
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

        # Process query
        response = query_legal_documents(
            client=client,
            query="Are there restrictions on drug paraphernalia sales?",
            retrieval_results=results
        )

        print(f"Answer: {response.short_answer}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Citations: {response.citations}")
    """
    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    # Validate inputs
    if not client:
        logger.error("Client is required for query processing")
        raise ValueError("Client is required for query processing")

    if not query or not query.strip():
        logger.error("Query cannot be empty for query processing")
        raise ValueError("Query cannot be empty for query processing")

    if not retrieval_results:
        logger.error("Retrieval results are required for query processing")
        raise ValueError("Retrieval results are required for query processing")

    logger.info(f"Processing query: '{query[:50]}...'")
    logger.debug(f"Using model: {model}, temperature: {temperature}")

    # Extract sections from retrieval results
    sections = retrieval_results.get("sections", [])
    if not sections:
        logger.warning("No sections found in retrieval results")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found.",
            reasoning="The search did not return any legal sections that address your query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available to answer the query.",
        )

    logger.info(f"Found {len(sections)} relevant sections to analyze")

    # Prepare context for the LLM
    context_sections = []
    for i, section in enumerate(sections):
        section_text = f"""
Section {i + 1}: {section.get("heading_text", "Untitled Section")}
Relevance Score: {section.get("relevance_score", 0):.3f}
Content: {section.get("body_text", "")}

Matching Segments:
"""
        # Add matching segments for context
        for j, segment in enumerate(
            section.get("matching_segments", [])[:3]
        ):  # Limit to top 3 segments
            segment_text = segment.get("segment_text", "")
            if segment_text:
                section_text += f"  - Segment {j + 1}: {segment_text}\n"

        context_sections.append(section_text)

    full_context = "\n".join(context_sections)

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
    queries: List[str],
    jurisdiction_id: str,
    sections_parquet_path: str,
    collection,
    model: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    n_results: int = 10,
    use_hyde: bool = False,
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

        # Run multiple queries
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
            model=Config.get_powerful_model()
        )

        # View results
        print(results_df.select(["query", "short_answer", "confidence"]))
    """
    import time

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

    # Use default model if not specified
    if model is None:
        model = Config.get_fast_model()

    logger.info(
        f"Processing {len(queries)} queries for jurisdiction: {jurisdiction_id}"
    )
    logger.debug(f"Using model: {model}, n_results: {n_results}, use_hyde: {use_hyde}")

    results = []

    for i, query in enumerate(queries):
        if query is None or not isinstance(query, str) or not query.strip():
            logger.warning(f"Skipping empty query at index {i}")
            continue

        start_time = time.time()
        logger.info(f"Processing query {i + 1}/{len(queries)}: '{query[:50]}...'")

        try:
            retrieval_results = retrieve_sections(
                collection=collection,
                query_text=query,
                sections_parquet_path=sections_parquet_path,
                n_results=n_results,
                jurisdiction_id=jurisdiction_id,
                rewrite=use_hyde,
                client=client if use_hyde else None,
                model=model,
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
            )

            processing_time = time.time() - start_time

            result = {
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

            results.append(result)

            logger.info(
                f"Query {i + 1} completed - confidence: {query_response.confidence:.2f}, "
                f"sections: {sections_found}, time: {processing_time:.2f}s"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query {i + 1} failed: {str(e)}")

            # Add failed result with error information
            result = {
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

            results.append(result)

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
