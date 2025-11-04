import marimo

__generated_with = "0.17.6"
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
    from legiscope.query import query_legal_documents, format_query_response
    import traceback


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## ChromaDB setup
    """)
    return


@app.cell
def _():
    collection_name = "legal_code_all"
    chroma_path = "./data/chroma_db"

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
        print("Check ChromaDB is set up with embedded legal documents.")
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
    embedding_client = None
    embedding_model = "embeddinggemma"

    try:
        import ollama

        embedding_client = ollama.Client()

        test_response = embedding_client.embeddings(model=embedding_model, prompt="test")
        if test_response and "embedding" in test_response:
            embedding_dim = len(test_response["embedding"])
            print(f"=== Embedding Client Setup ===")
            print(f"Client: ollama")
            print(f"Model: {embedding_model}")
            print(f"Dimension: {embedding_dim}")
            print("Client setup successful")
        else:
            print("ERROR: Embedding client test failed")
            embedding_client = None

    except ImportError:
        print("ERROR: ollama package not available")
        print("Install with: uv pip install ollama")
    except Exception as e:
        print(f"ERROR: Embedding client setup failed: {str(e)}")
        embedding_client = None
    return


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
    sections_parquet_path = "./data/laws/IL-WindyCity/tables/sections.parquet"

    print("=== Query Configuration ===")
    print(f"Query: {query}")
    print(f"Max results: {n_results}")
    print(f"HYDE rewriting: {'Enabled' if use_hyde else 'Disabled'}")
    print(f"Sections file: {sections_parquet_path}")
    return jurisdiction_id, n_results, query, sections_parquet_path, use_hyde


@app.cell
def _():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    instructor_client = None

    print("=== OpenAI Client Setup ===")
    print(f"OpenAI API Key found: {'Yes' if openai_api_key else 'No'}")

    if openai_api_key:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            instructor_client = instructor.from_openai(openai_client)
            print("Instructor client created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create instructor client: {str(e)}")
            instructor_client = None
    else:
        print("ERROR: OpenAI API key not found in environment")
        print("Set OPENAI_API_KEY environment variable to use query processing")
    return (instructor_client,)


@app.cell
def _(
    collection,
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

            results = retrieve_sections(
                collection=collection,
                query_text=query,
                sections_parquet_path=sections_parquet_path,
                n_results=n_results,
                jurisdiction_id=jurisdiction_id,
                rewrite=use_hyde,
                client=instructor_client if use_hyde else None,
                model="gpt-4.1-mini",
            )

            print(f"Raw results structure: {list(results.keys()) if results else 'None'}")

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
def _():
    print("=== Query Processing Setup ===")
    print("Query processing functions available from setup imports")
    query_processing_available = True
    return (query_processing_available,)


@app.cell
def _(instructor_client, query_processing_available, results):
    query_response = None

    print("=== Query Processing ===")
    print(f"Query processing available: {'Yes' if query_processing_available else 'No'}")
    print(
        f"Instructor client available: {'Yes' if instructor_client is not None else 'No'}"
    )
    print(f"Results available: {'Yes' if results is not None else 'No'}")

    if (
        query_processing_available
        and instructor_client is not None
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
                model="gpt-4.1",  # Use more powerful model as requested
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


@app.cell
def _(query_response):
    print("=== Query Response ===")

    if query_response is None:
        print("No query response to display")
    else:
        # Format and display the response
        formatted_response = format_query_response(query_response)
        print(formatted_response)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
