import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import os
    import chromadb

    import sys

    # Add src to path to import legiscope modules
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Note: Load environment variables before running this notebook:
    #   export $(cat ../.env | grep -v '^#' | xargs)
    #   uv run marimo edit notebooks/query_demo.py

    from legiscope.retrieve import (
        SectionRetrievalSettings,
        retrieve_sections,
        get_jurisdiction_stats,
        filter_sections,
    )
    from legiscope.llm_config import Config
    from legiscope.utils import LLMConfig
    from legiscope.query import (
        QuerySettings,
        query_legal_documents,
        format_query_response,
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ChromaDB setup

    We use a different collection name for different embedding models. In particular,
    - `legal_code_ollama` holds vectors created with Google's `embeddinggemma` model, running locally on ollama
    - `legal_code_mistral` holds vectors created with Mistral AI's `mistral-embed` model, running on Mistral AI's cloud platform.

    The collection name can be configured via the `LEGISCOPE_COLLECTION_NAME` environment variable.
    """)
    return


@app.cell
def _():
    # Use environment variable for collection name, default to legal_code_all
    collection_name = os.getenv("LEGISCOPE_COLLECTION_NAME", "legal_code_all")
    # Dynamically resolve path relative to this file
    chroma_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    print("=== ChromaDB Overview ===")
    print(f"Collection: {collection_name}")
    print(f"Path: {chroma_path}")
    print(f"Collection object: {collection}")

    stats = get_jurisdiction_stats(collection)
    assert stats is not None

    print(f"Total Documents: {stats.total_documents}")
    print(f"Jurisdictions: {len(stats.jurisdictions)}")
    print(f"States: {len(stats.states)}")
    return (collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration Summary

    This notebook uses the following environment variables for configuration:

    **LLM Configuration:**
    - `LEGISCOPE_LLM_PROVIDER`: LLM provider ("openai" or "mistral")
    - `LEGISCOPE_FAST_MODEL`: Fast model name (overrides provider default)
    - `LEGISCOPE_POWERFUL_MODEL`: Powerful model name (overrides provider default)

    **Embedding Configuration:**
    - `LEGISCOPE_EMBEDDING_PROVIDER`: Embedding provider ("ollama" or "mistral")
    - `LEGISCOPE_EMBEDDING_MODEL`: Embedding model name

    **Database Configuration:**
    - `LEGISCOPE_COLLECTION_NAME`: ChromaDB collection name

    **Query Configuration:**
    - `LEGISCOPE_N_RESULTS`: Number of results to retrieve (default: 10)
    - `LEGISCOPE_USE_HYDE`: Enable HYDE query rewriting ("true" or "false", default: false)
    """)
    return


@app.cell
def _():
    from legiscope.embeddings import get_embedding_client, get_embeddings

    embedding_client = None
    # Use environment variables for embedding configuration
    embedding_provider = os.getenv("LEGISCOPE_EMBEDDING_PROVIDER", "mistral")
    embedding_model = os.getenv("LEGISCOPE_EMBEDDING_MODEL", "mistral-embed")

    embedding_client = get_embedding_client(embedding_provider)

    test_response = get_embeddings(
        embedding_client,
        ["test"],
        model=embedding_model,
        provider=embedding_provider,
    )
    assert test_response is not None and len(test_response) > 0

    embedding_dim = len(test_response[0])
    print("=== Embedding Client Setup ===")
    print(f"Client: {embedding_provider}")
    print(f"Model: {embedding_model}")
    print(f"Dimension: {embedding_dim}")
    return embedding_client, embedding_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieval
    """)
    return


@app.cell
def _():
    query = (
        "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?"
    )

    # Search parameters (configurable via environment variables)
    n_results = int(os.getenv("LEGISCOPE_N_RESULTS", "10"))
    use_hyde = os.getenv("LEGISCOPE_USE_HYDE", "false").lower() == "true"

    jurisdiction_id = "CA-LosAngeles"

    # Sections parquet path for full section context (constructed from jurisdiction)
    jurisdiction_path = "CA-LosAngeles"
    sections_parquet_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "laws",
        jurisdiction_path,
        "tables",
        "sections.parquet",
    )

    print("=== Query Configuration ===")
    print(f"Query: {query}")
    print(f"Max results: {n_results}")
    print(f"HYDE rewriting: {'Enabled' if use_hyde else 'Disabled'}")
    print(f"Jurisdiction ID: {jurisdiction_id}")
    print(f"Sections file: {sections_parquet_path}")
    return jurisdiction_id, n_results, query, sections_parquet_path, use_hyde


@app.cell
def _():
    print("=== LLM Client Setup ===")
    print(f"Using instructor with {Config.get_llm_provider()} provider")
    print(f"Fast model: {Config.get_fast_model()}")
    print(f"Powerful model: {Config.get_powerful_model()}")

    instructor_client = None
    instructor_client = Config.get_fast_client()
    assert instructor_client is not None, "Failed to initialize instructor client"
    return (instructor_client,)


@app.cell
def _(
    collection,
    embedding_client,
    embedding_model,
    instructor_client,
    jurisdiction_id,
    n_results,
    query,
    sections_parquet_path,
    use_hyde,
):
    results = None
    sections = []

    # Create settings for section retrieval
    settings = SectionRetrievalSettings(
        n_results=n_results,
        jurisdiction_id=jurisdiction_id,
        use_hyde=use_hyde,
        hyde_client=instructor_client if use_hyde else None,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )

    results = retrieve_sections(
        collection=collection,
        sections_parquet_path=sections_parquet_path,
        query_text=query,
        settings=settings,
    )

    if results and results.sections:
        sections = results.sections
        result_count = len(sections)
        print(f"Number of sections found: {result_count}")

        # Show query info
        query_info = results.query_info
        print(f"Total segments found: {query_info.total_segments_found}")
        print(f"Unique sections: {query_info.unique_sections}")

        if use_hyde and instructor_client:
            print("Query rewriting: HYDE applied")
    else:
        print("WARNING: no results found or no matching sections")
    return query_info, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filter retrieved sections for relevance
    """)
    return


@app.cell
def _(instructor_client, query, results):
    # Filter section results using LLM-powered relevance assessment
    assert instructor_client is not None and results is not None
    filtered_results = filter_sections(
        client=instructor_client,
        sections_results=results,
        query=query,
        confidence_threshold=0.5,
    )

    # Show filtering statistics
    print("=== LLM-Powered Relevance Filtering ===")

    print(f"Original results: {filtered_results.filtering_metadata.original_count}")
    print(f"Filtered results: {filtered_results.filtering_metadata.filtered_count}")

    # Show relevance scores for filtered sections
    filtered_sections = filtered_results.sections
    if filtered_sections:
        print("\nRelevance scores of filtered sections:")
        for _i, _section in enumerate(filtered_sections):
            _score = _section.relevance_score
            _heading = _section.heading_text[:50]
            print(
                f"  {_i + 1}. {_score:.3f} - {_heading}{'...' if len(_heading) >= 50 else ''}"
            )
    return filtered_results, filtered_sections


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Show retrieval results
    """)
    return


@app.cell
def _(filtered_results, filtered_sections, query_info):
    print("=== Results Display ===")

    if filtered_results is None:
        print("No results to display")
    elif not filtered_results.sections:
        print("No matching sections found")
    else:
        print(f"Retrieval Results - Found {len(filtered_sections)} sections")
        print(f"From {query_info.total_segments_found} total matching segments")

        # Display each section result
        for i, result_section in enumerate(filtered_sections):
            relevance_score = result_section.relevance_score
            segment_count = result_section.segment_count

            print(
                f"\n--- Section {i + 1} (Relevance: {relevance_score:.3f}, {segment_count} matching segments) ---"
            )

            heading = result_section.heading_text
            print(f"Heading: {heading}")

            body_text = result_section.body_text
            if body_text:
                body_preview = (
                    body_text[:300] + "..." if len(body_text) > 300 else body_text
                )
                print(f"Content: {body_preview}")
            else:
                print("Content: [No body content]")

            matching_segments = result_section.matching_segments
            if matching_segments:
                print(f"Matching segments: {len(matching_segments)}")
                # Show first matching segment as preview
                if matching_segments:
                    first_segment = matching_segments[0]
                    segment_text = first_segment.segment_text
                    segment_preview = (
                        segment_text[:150] + "..."
                        if len(segment_text) > 150
                        else segment_text
                    )
                    print(f"Best match: {segment_preview}")

            print("---")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding client setup
    """)
    return


@app.cell
def _():
    print("=== Current Configuration ===")
    print(f"LLM Provider: {Config.get_llm_provider()}")
    print(f"Fast Model: {Config.get_fast_model()}")
    print(f"Powerful Model: {Config.get_powerful_model()}")
    print(f"Embedding Provider: {os.getenv('LEGISCOPE_EMBEDDING_PROVIDER', 'mistral')}")
    print(f"Embedding Model: {os.getenv('LEGISCOPE_EMBEDDING_MODEL', 'mistral-embed')}")
    print(
        f"Collection Name: {os.getenv('LEGISCOPE_COLLECTION_NAME', 'legal_code_mistral')}"
    )
    return


@app.cell
def _(filtered_results, instructor_client, query):
    query_response = None
    assert (
        instructor_client is not None
        and filtered_results is not None
        and filtered_results.sections
    )

    print("=== Query Processing ===")

    # Create query settings
    llm_config = LLMConfig(client=instructor_client, temperature=0.0, max_retries=3)
    query_settings = QuerySettings(llm=llm_config)

    query_response = query_legal_documents(
        retrieval_results=filtered_results,
        query=query,
        settings=query_settings,
    )

    print("Query processing completed successfully")
    print(f"Answer confidence: {query_response.confidence:.1%}")
    print(f"Number of citations: {len(query_response.citations)}")
    print(f"Number of supporting passages: {len(query_response.supporting_passages)}")
    return (query_response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query Response
    """)
    return


@app.cell
def _(mo, query_response):
    # Use the imported format_query_response function
    formatted_response = format_query_response(query_response)
    mo.md(formatted_response)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
