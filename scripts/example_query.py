#!/usr/bin/env python3
"""
Example script demonstrating the query functionality.

This script shows how to use the query module to process user queries
against retrieved legal documents.
"""

import os
import sys
from pathlib import Path

# Add src to path to import legiscope modules
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import instructor
from openai import OpenAI
from legiscope.query import query_legal_documents, format_query_response, run_queries


def example_query_processing():
    """Demonstrate query processing with sample data."""

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable to run this example."
        )
        print(
            "This example demonstrates the structure without making actual API calls."
        )

        # Show what response structure would look like
        from legiscope.query import LegalQueryResponse

        sample_response = LegalQueryResponse(
            short_answer="Yes, the jurisdiction has laws restricting the sale of drug paraphernalia.",
            reasoning="The municipal code explicitly prohibits the sale, distribution, or possession with intent to sell drug paraphernalia. The law defines drug paraphernalia broadly and establishes penalties for violations.",
            citations=["Drug Paraphernalia Regulations", "Penalties for Violations"],
            supporting_passages=[
                "No person shall sell, distribute, or possess with intent to sell drug paraphernalia.",
                "Any person convicted of violating the drug paraphernalia regulations shall be subject to a fine of not more than $1,000, imprisonment for not more than 6 months, or both.",
            ],
            confidence=0.95,
            limitations="Based on the provided sample municipal code sections only.",
        )

        print("\n=== Sample Single Query Response ===")
        print(format_query_response(sample_response))

        # Demonstrate run_queries function structure
        print("\n=== Sample Multiple Queries Structure ===")
        print("With OpenAI API key, you could run:")
        print("""
queries = [
    "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?",
    "What are the parking regulations?",
    "Do I need a permit for a home business?"
]

results_df = run_queries(
    client=client,
    queries=queries,
    jurisdiction_id="IL-WindyCity",
    sections_parquet_path="./data/laws/IL-WindyCity/tables/sections.parquet",
    collection=collection,
    model="gpt-4.1"
)

print(results_df.select(["query", "short_answer", "confidence"]))
        """)
        return

    client = instructor.from_openai(OpenAI())

    print("=== Example 1: Single Query Processing ===")

    sample_results = {
        "sections": [
            {
                "section_idx": 1,
                "heading_text": "Drug Paraphernalia Regulations",
                "body_text": "No person shall sell, distribute, or possess with intent to sell drug paraphernalia. Drug paraphernalia includes any equipment, product, or material that is intended for use in manufacturing, compounding, converting, concealing, producing, processing, preparing, injecting, ingesting, inhaling, or otherwise introducing into the human body a controlled substance.",
                "relevance_score": 0.05,
                "matching_segments": [
                    {
                        "segment_idx": 1,
                        "segment_text": "No person shall sell, distribute, or possess with intent to sell drug paraphernalia.",
                        "distance": 0.05,
                        "segment_position": 0,
                    }
                ],
            }
        ],
        "query_info": {
            "original_query": "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?",
            "total_segments_found": 1,
            "unique_sections": 1,
        },
    }

    query = (
        "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?"
    )

    try:
        print(f"Processing query: '{query}'")
        print(f"Found {len(sample_results['sections'])} relevant sections")

        response = query_legal_documents(
            client=client,
            query=query,
            retrieval_results=sample_results,
            model="gpt-4.1",  # Use more powerful model as requested
            temperature=0.1,
            max_retries=3,
        )

        print("\n=== Single Query Response ===")
        print(format_query_response(response))

    except Exception as e:
        print(f"Error processing single query: {e}")

    print("\n=== Example 2: Multiple Queries Processing ===")
    print("This example requires ChromaDB setup with actual legal data.")
    print("See demo_nb.py for complete ChromaDB setup example.")

    # Show structure of multiple queries
    queries = [
        "Does the jurisdiction have laws that restrict the sale of drug paraphernalia?",
        "What are the parking regulations?",
        "Do I need a permit for a home business?",
    ]

    print(f"Sample queries to process: {len(queries)}")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    print("\nTo run these queries, you would need:")
    print("1. ChromaDB collection with embedded legal documents")
    print("2. Sections parquet file for the jurisdiction")
    print("3. Then call run_queries() as shown in the function documentation")


if __name__ == "__main__":
    example_query_processing()
