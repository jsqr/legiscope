import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    from pydantic import BaseModel
    import os
    import chromadb
    from pathlib import Path

    import instructor
    from openai import OpenAI
    import sys
    import os
    import traceback

    # Add src to path to import legiscope modules
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from legiscope.retrieve import (
        retrieve_embeddings,
        retrieve_sections,
        get_jurisdiction_stats,
    )
    from legiscope.utils import ask
    from legiscope.llm_config import Config
    from legiscope.query import query_legal_documents, format_query_response


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
    """)
    return


@app.cell
def _():
    collection_name = "legal_code_mistral"
    # collection_name = "legal_code_ollama"
    chroma_path = "../data/chroma_db"

    try:
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        collection = chroma_client.get_or_create_collection(name=collection_name)

        print("=== ChromaDB Overview ===")
        print(f"Collection: {collection_name}")
        print(f"Path: {chroma_path}")
        print(f"Collection object: {collection}")

        stats = get_jurisdiction_stats(collection)

        print(f"Stats: {stats}")

        if stats:
            print(f"Total Documents: {stats.get('total_documents', 0)}")
            print(f"Jurisdictions: {len(stats.get('jurisdictions', {}))}")
            print(f"States: {len(stats.get('states', {}))}")
            print("ChromaDB Connected Successfully")
        else:
            print("WARNING: Collection Connected but No Data Found")
            print("The collection exists but contains no embedded documents.")

    except Exception as e:
        print(f"ERROR: ChromaDB connection failed")
        print(f"Error: {str(e)}")
        print("Check ChromaDB is set up with embedded documents.")
        collection = None
        chroma_client = None
    return (collection,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Embedding client setup
    """)
    return


@app.cell
def _():
    from legiscope.embeddings import get_embedding_client, get_embeddings

    embedding_client = None
    embedding_model = "mistral-embed"
    embedding_provider = "mistral"

    try:
        embedding_client = get_embedding_client(embedding_provider)

        test_response = get_embeddings(
            embedding_client,
            ["test"],
            model=embedding_model,
            provider=embedding_provider,
        )
        if test_response and len(test_response) > 0:
            embedding_dim = len(test_response[0])
            print(f"=== Embedding Client Setup ===")
            print(f"Client: {embedding_provider}")
            print(f"Model: {embedding_model}")
            print(f"Dimension: {embedding_dim}")
            print("Client setup successful")
        else:
            print("ERROR: Embedding client test failed")
            embedding_client = None

    except Exception as e:
        print(f"ERROR: Embedding client setup failed: {str(e)}")
        embedding_client = None
    return embedding_client, embedding_model


@app.cell
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

    # Search parameters
    n_results = 10
    use_hyde = False  # Disabled for debugging

    # Optional jurisdiction filters
    jurisdiction_id = "IL-WindyCity"
    # state = "IL"  # All jurisdictions in a state
    # municipality = "WindyCity"  # Specific municipality

    # Sections parquet path for full section context
    sections_parquet_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "laws",
        "IL-WindyCity",
        "tables",
        "sections.parquet",
    )

    print("=== Query Configuration ===")
    print(f"Query: {query}")
    print(f"Max results: {n_results}")
    print(f"HYDE rewriting: {'Enabled' if use_hyde else 'Disabled'}")
    print(f"Sections file: {sections_parquet_path}")
    return jurisdiction_id, n_results, query, sections_parquet_path, use_hyde


@app.cell
def _():
    instructor_client = None

    print("=== LLM Client Setup ===")
    print("Using instructor with Mistral provider")

    try:
        instructor_client = Config.get_default_client()
        print(f"Instructor client created with model {instructor_client}")
    except Exception as e:
        print(f"ERROR: Failed to create instructor client: {str(e)}")
        instructor_client = None
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

    print("=== Retrieval ===")
    print(f"ChromaDB collection available: {'Yes' if collection is not None else 'No'}")
    print(
        f"Instructor client available: {'Yes' if instructor_client is not None else 'No'}"
    )
    print(f"Query: {query}")
    print(f"Using HYDE: {use_hyde}")

    if collection is None:
        print("ERROR: Cannot execute retrieval")
        print("ChromaDB collection is not available.")
    else:
        try:
            print("Executing retrieval...")

            # Use the existing embedding client from setup
            results = retrieve_sections(
                collection=collection,
                query_text=query,
                sections_parquet_path=sections_parquet_path,
                n_results=n_results,
                jurisdiction_id=jurisdiction_id,
                rewrite=use_hyde,
                client=instructor_client if use_hyde else None,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
            )

            print(
                f"Raw results structure: {list(results.keys()) if results else 'None'}"
            )

            if results and results.get("sections"):
                sections = results["sections"]
                result_count = len(sections)
                print(f"Retrieval done")
                print(f"Number of sections found: {result_count}")

                # Show query info
                query_info = results.get("query_info", {})
                print(
                    f"Total segments found: {query_info.get('total_segments_found', 0)}"
                )
                print(f"Unique sections: {query_info.get('unique_sections', 0)}")

                if use_hyde and instructor_client:
                    print("Query rewriting: HYDE applied")
            else:
                print("WARNING: No Results Found")
                print("No matching sections were found for the query.")

        except Exception as e:
            print(f"ERROR: Retrieval failed")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            results = None
            sections = []
    return results, sections


@app.cell
def _(mo):
    mo.md(r"""
    ## Show retrieval results
    """)
    return


@app.cell
def _(results, sections):
    print("=== Results Display ===")

    if results is None:
        print("No results to display")
    elif not results.get("sections"):
        print("No matching sections found")
    else:
        _sections = results["sections"]
        _query_info = results.get("query_info", {})

        print(f"Retrieval Results - Found {len(sections)} sections")
        print(
            f"From {_query_info.get('total_segments_found', 0)} total matching segments"
        )

        # Display each section result
        for i, section in enumerate(_sections):
            relevance_score = section.get("relevance_score", 0)
            segment_count = section.get("segment_count", 0)

            print(
                f"\n--- Section {i + 1} (Relevance: {relevance_score:.3f}, {segment_count} matching segments) ---"
            )

            # Display section heading
            heading = section.get("heading_text", "No heading")
            print(f"Heading: {heading}")

            # Display section body (truncated)
            body_text = section.get("body_text", "")
            if body_text:
                body_preview = (
                    body_text[:300] + "..." if len(body_text) > 300 else body_text
                )
                print(f"Content: {body_preview}")
            else:
                print("Content: [No body content]")

            # Display matching segments info
            matching_segments = section.get("matching_segments", [])
            if matching_segments:
                print(f"Matching segments: {len(matching_segments)}")
                # Show first matching segment as preview
                if matching_segments:
                    first_segment = matching_segments[0]
                    segment_text = first_segment.get("segment_text", "")
                    segment_preview = (
                        segment_text[:150] + "..."
                        if len(segment_text) > 150
                        else segment_text
                    )
                    print(f"Best match: {segment_preview}")

            print("---")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Query Processing
    """)
    return


@app.cell
def _(instructor_client, query_processing_available, results):
    query_response = None

    print("=== Query Processing ===")
    print(
        f"Instructor client available: {'Yes' if instructor_client is not None else 'No'}"
    )
    print(f"Results available: {'Yes' if results is not None else 'No'}")

    if (
        instructor_client is not None
        and results is not None
        and results.get("sections")
    ):
        try:
            print("Processing query with LLM analysis...")

            # Use the same query from the retrieval step
            user_query = "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?"

            query_response = query_legal_documents(
                client=instructor_client,
                query=user_query,
                retrieval_results=results,
                temperature=0.1,
                max_retries=3,
            )

            print("Query processing completed successfully")
            print(f"Answer confidence: {query_response.confidence:.1%}")
            print(f"Number of citations: {len(query_response.citations)}")
            print(
                f"Number of supporting passages: {len(query_response.supporting_passages)}"
            )

        except Exception as e:
            print(f"ERROR: Query processing failed")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            query_response = None
    else:
        print("Cannot process query - missing requirements")
        if not query_processing_available:
            print("  - Query processing functions not available")
        if instructor_client is None:
            print("  - Instructor client not available")
        if results is None:
            print("  - No retrieval results available")
        elif not results.get("sections"):
            print("  - No sections in retrieval results")
    return (query_response,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Query Response
    """)
    return


@app.function
def format_query_response_md(response):
    md_output = f"## Query Response\n\n"
    md_output += f"**Answer:** {response.short_answer}\n\n"
    md_output += f"**Confidence:** {response.confidence:.1%}\n\n"

    # Add citations if available
    if response.citations:
        md_output += "### Citations\n"
        for i, citation in enumerate(response.citations, 1):
            md_output += f"{i}. {citation}\n"
        md_output += "\n"

    # Add supporting passages if available
    if response.supporting_passages:
        md_output += "### Supporting Passages\n"
        for i, passage in enumerate(response.supporting_passages, 1):
            md_output += f"{i}. {passage}\n"
        md_output += "\n"

    return md_output


@app.cell
def _(mo, query_response):
    formatted_response = format_query_response_md(query_response)
    mo.md(formatted_response)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
