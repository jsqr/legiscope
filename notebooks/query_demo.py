import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    from legiscope.models import JurisdictionRef, CodeRef
    from legiscope.embeddings import (
        get_embedding_client,
        get_embeddings,
        get_or_create_legal_collection,
    )
    from legiscope.retrieve import (
        SectionRetrievalSettings,
        retrieve_sections,
        get_jurisdiction_stats,
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


@app.cell
def _():
    import os
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent
    os.chdir(_project_root)
    project_root = _project_root
    return (project_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Legal Code Query Demo

    This notebook demonstrates the legiscope retrieval and query pipeline:
    1. Connect to ChromaDB and verify the jurisdiction has been indexed
    2. Retrieve relevant sections via semantic search
    3. Query the LLM for a structured legal analysis

    Configuration is read from `params.yaml` and `config.yaml`.
    API keys should be set in the environment (`OPENAI_API_KEY`, `MISTRAL_API_KEY`).
    """)
    return


@app.cell
def _(project_root):
    # --- Jurisdiction / code to query ---
    code_ref = CodeRef(
        jurisdiction=JurisdictionRef(state="IL", locality="WindyCity"),
        code_slug="municipal-code",
    )

    sections_parquet = code_ref.full_data_dir / "sections.parquet"
    assert sections_parquet.exists(), (
        f"sections.parquet not found at {sections_parquet} — "
        "run the pipeline first (./scripts/dvc_repro.sh)"
    )

    print(f"Jurisdiction: {code_ref.jurisdiction_id}")
    print(f"Code: {code_ref.code_slug}")
    print(f"Data dir: {code_ref.full_data_dir}")
    return code_ref, sections_parquet


@app.cell
def _(project_root):
    from legiscope.embeddings import CollectionConfig, EMBEDDING_PROVIDER

    collection = get_or_create_legal_collection(
        CollectionConfig(provider=EMBEDDING_PROVIDER)
    )
    stats = get_jurisdiction_stats(collection)

    print(f"Collection: {collection.name}")
    print(f"Total documents: {stats.total_documents}")
    print(f"Jurisdictions: {len(stats.jurisdictions)}")
    return (collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieval
    """)
    return


@app.cell
def _(code_ref, collection, sections_parquet):
    query = "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?"

    embedding_client = get_embedding_client()

    settings = SectionRetrievalSettings(
        jurisdiction_id=code_ref.jurisdiction_id,
        embedding_client=embedding_client,
    )

    results = retrieve_sections(
        collection=collection,
        sections_parquet_path=str(sections_parquet),
        query_text=query,
        settings=settings,
    )

    print(f"Query: {query}")
    print(f"Sections found: {len(results.sections)}")
    print(f"Total segments: {results.query_info.total_segments_found}")
    return query, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieval results
    """)
    return


@app.cell
def _(results):
    for i, s in enumerate(results.sections):
        heading = s.heading_text[:80]
        print(f"  {i+1}. [{s.relevance_score:.3f}] {heading}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM query
    """)
    return


@app.cell
def _(query, results):
    llm_config = LLMConfig(client=Config.get_powerful_client(), temperature=0.0)
    query_settings = QuerySettings(
        llm=llm_config,
        filter_relevance=True,
        relevance_threshold=0.5,
    )

    response, scores = query_legal_documents(
        retrieval_results=results,
        query=query,
        settings=query_settings,
    )

    print(f"Confidence: {response.confidence:.1%}")
    print(f"Citations: {len(response.citations)}")
    print(f"Supporting passages: {len(response.supporting_passages)}")
    return (response,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query Response
    """)
    return


@app.cell
def _(mo, response):
    mo.md(format_query_response(response))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
