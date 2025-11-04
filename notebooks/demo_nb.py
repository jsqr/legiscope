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
    from legiscope.retrieve import retrieve_embeddings, get_jurisdiction_stats
    from legiscope.utils import ask


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

    print("=== Query Configuration ===")
    print(f"Query: {query}")
    print(f"Max results: {n_results}")
    print(f"HYDE rewriting: {'Enabled' if use_hyde else 'Disabled'}")
    return n_results, query, use_hyde


@app.cell
def _():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    instructor_client = None

    print("=== OpenAI Client Setup ===")
    print(f"OpenAI API Key found: {'Yes' if openai_api_key else 'No'}")
    return (instructor_client,)


@app.cell
def _(
    collection,
    embedding_client,
    embedding_model,
    instructor_client,
    n_results,
    query,
    use_hyde,
):
    results = None

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

            results = retrieve_embeddings(
                collection=collection,
                query_text=query,
                n_results=n_results,
                rewrite=use_hyde,
                client=instructor_client if use_hyde else None,
                model="gpt-4.1-mini",
                embedding_client=embedding_client,
                embedding_model=embedding_model,
            )

            print(f"Raw results: {results}")

            if results and results.get("ids") and results["ids"][0]:
                result_count = len(results["ids"][0])
                print(f"Retrieval done")
                print(f"Number of results found: {result_count}")

                if use_hyde and instructor_client:
                    print("Query rewriting: HYDE applied")
            else:
                print("WARNING: No Results Found")
                print("No matching documents were found for the query.")

        except Exception as e:
            print(f"ERROR: Retrieval failed")
            print(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()
            results = None
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Show retrieval results
    """)
    return


@app.cell
def _(results):
    print("=== Results ===")

    if results is None:
        print("No results to display")
    elif not results.get("ids") or not results["ids"][0]:
        print("No matching documents")
    else:
        print(f"Retrieval results - found {len(results['ids'][0])} documents")

        # Extract result data
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [None])[0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        # Display each result
        for i, (doc, metadata, distance, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            print(f"\n--- Result {i + 1} (Distance: {distance:.3f}) ---")

            # Display document content (truncated)
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"Content: {doc_preview}")

            # Display metadata if available
            if metadata:
                metadata_lines = []
                if metadata.get("jurisdiction_id"):
                    metadata_lines.append(f"Jurisdiction: {metadata['jurisdiction_id']}")
                if metadata.get("state"):
                    metadata_lines.append(f"State: {metadata['state']}")
                if metadata.get("municipality"):
                    metadata_lines.append(f"Municipality: {metadata['municipality']}")
                if metadata.get("section_heading"):
                    metadata_lines.append(f"Section: {metadata['section_heading']}")

                if metadata_lines:
                    print(" | ".join(metadata_lines))

            print("---")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
