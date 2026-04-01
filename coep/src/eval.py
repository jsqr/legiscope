"""
Evaluation module for assessing the quality of legal query responses.
This module implements LLM-as-a-judge patterns to score generated answers
against ground truth human-authored answers.
"""

from typing import Literal
from pydantic import BaseModel, Field
from polars import DataFrame
import polars as pl
from loguru import logger
from legiscope.utils import LLMConfig


class EvaluationResult(BaseModel):
    """Structured output for the evaluation of a single query response."""

    score: int = Field(
        ...,
        description="A score from 0 to 10 evaluating the quality of the answer based on the ground truth.",
        ge=0,
        le=10,
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation for the score, citing what was correct, missing, or hallucinated.",
    )
    accuracy_label: Literal["Correct", "Partially Correct", "Incorrect"] = Field(
        ..., description="Categorical label for the accuracy of the response."
    )


class Evaluator:
    """Handles the evaluation of generated responses against ground truth."""

    def __init__(self, llm_config: LLMConfig | None = None):
        """
        Initialize the evaluator.

        Args:
            llm_config: Configuration for the judge LLM. If None, uses the powerful client.
        """
        if llm_config is None:
            from legiscope.llm_config import Config

            # We want a powerful model for evaluation (Judge)
            # Config.get_powerful_client() already returns an Instructor client
            self.client = Config.get_powerful_client()
            self._request_params = Config.get_llm_params()
        else:
            # llm_config.client is already an Instructor client
            self.client = llm_config.client
            self._request_params = {
                "temperature": llm_config.temperature,
                "max_retries": llm_config.max_retries,
            }

    def evaluate_response(
        self, question: str, generated_answer: str, ground_truth: str
    ) -> EvaluationResult:
        """
        Evaluate a single generated response against the ground truth using the LLM judge.
        """
        system_prompt = (
            "You are an expert legal scholar and evaluator. "
            "Your task is to grade the quality of an AI-generated legal answer compared to a known correct human-written answer."
        )

        user_content = f"""
        Question: {question}

        Ground Truth Answer: {ground_truth}

        Generated Answer: {generated_answer}

        Evaluate the Generated Answer based on:
        1. Accuracy: Does it convey the same legal facts as the ground truth?
        2. Completeness: Does it cover all aspects mentioned in the ground truth?
        3. Hallucinations: Does it include false information not present in the ground truth?

        Assign a score from 0 to 10, where:
        - 10: Perfect match in facts and nuance.
        - 7-9: Mostly correct, minor details missing or slightly different phrasing.
        - 4-6: Partially correct, misses key points or has minor inaccuracies.
        - 0-3: Completely wrong, irrelevant, or dangerous hallucination.

        Important notes:
        - The ground truth may contain binary results, where 0 = "No" and 1 = "Yes". 
          The generated answer may use different phrasing (e.g., "No", "False", "0", "Negative" for 0). 
          Focus on the meaning rather than exact wording.
        """

        try:
            return self.client.chat.completions.create(
                response_model=EvaluationResult,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                **self._request_params,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                score=0,
                reasoning=f"Evaluation execution failed: {e}",
                accuracy_label="Incorrect",
            )

    def evaluate_batch(
        self, df: DataFrame, question_col: str, answer_col: str, truth_col: str
    ) -> DataFrame:
        """
        Run evaluation on a DataFrame containing results and ground truth.
        """
        scores = []
        reasons = []
        labels = []

        logger.info(f"Starting evaluation of {len(df)} records...")

        for row in df.iter_rows(named=True):
            result = self.evaluate_response(
                question=row[question_col],
                generated_answer=row[answer_col],
                ground_truth=row[truth_col],
            )
            scores.append(result.score)
            reasons.append(result.reasoning)
            labels.append(result.accuracy_label)

        # Add evaluation columns to the dataframe
        return df.with_columns(
            [
                pl.Series("eval_score", scores),
                pl.Series("eval_reason", reasons),
                pl.Series("eval_label", labels),
            ]
        )


def load_and_filter_monqcle(
    monqcle_path: str,
    jurisdiction_name: str,
    series_title: str = "DPL_2025_Consolidated",
) -> pl.DataFrame:
    """
    Load MonQcle Standard Report and filter to target jurisdiction.

    The MonQcle data is in wide format (one row per jurisdiction, variables as columns).
    This function filters to the target jurisdiction and series.

    Args:
        monqcle_path: Path to MonQcle Standard Report CSV
        jurisdiction_name: Full jurisdiction name (e.g., "Los Angeles, Los Angeles County, California, United States")
        series_title: Series to filter on (default: DPL_2025_Consolidated)

    Returns:
        DataFrame with single row for target jurisdiction
    """
    df = pl.read_csv(monqcle_path)

    # Filter to target jurisdiction and series
    filtered = df.filter(
        (pl.col("name") == jurisdiction_name) & (pl.col("series_title") == series_title)
    )

    if len(filtered) == 0:
        available = (
            df.filter(pl.col("name").str.contains("Los Angeles"))["name"]
            .unique()
            .to_list()
        )
        raise ValueError(
            f"No records found for jurisdiction '{jurisdiction_name}' with series '{series_title}'. "
            f"Available Los Angeles jurisdictions: {available}"
        )

    if len(filtered) > 1:
        logger.warning(
            f"Multiple records found for {jurisdiction_name}, using first one"
        )
        filtered = filtered.head(1)

    logger.info(f"Found MonQcle record for {jurisdiction_name}")
    return filtered


def melt_monqcle_to_long(
    monqcle_row: pl.DataFrame, variable_names: list[str]
) -> pl.DataFrame:
    """
    Transform wide-format MonQcle data to long format for joining with query results.

    The MonQcle data has variables as columns (e.g., dp_law, dp_type, etc.).
    This melts it to long format with columns: variable_name, ground_truth

    Args:
        monqcle_row: Single-row DataFrame from MonQcle
        variable_names: List of variable names to extract (from queries file)

    Returns:
        DataFrame with variable_name and ground_truth columns
    """
    # Get the columns that exist in the MonQcle data
    available_cols = set(monqcle_row.columns)

    # Filter to only variables that exist in the data
    valid_variables = [v for v in variable_names if v in available_cols]
    missing_variables = [v for v in variable_names if v not in available_cols]

    if missing_variables:
        logger.warning(f"Variables not found in MonQcle data: {missing_variables}")

    if not valid_variables:
        raise ValueError("No valid variable names found in MonQcle data")

    # Extract values for each variable
    records = []
    row_dict = monqcle_row.to_dicts()[0]

    for var_name in valid_variables:
        value = row_dict.get(var_name, None)
        # Convert MonQcle's "-" placeholder to None/empty
        if value == "-" or value is None:
            ground_truth = ""
        else:
            ground_truth = str(value)
        records.append({"variable_name": var_name, "ground_truth": ground_truth})

    result = pl.DataFrame(records)
    logger.info(f"Melted {len(result)} variables to long format")
    return result


def jurisdiction_id_to_monqcle_name(jurisdiction_id: str) -> str:
    """
    Convert jurisdiction ID (e.g., CA-LosAngeles) to MonQcle name format.

    Currently hardcoded for known jurisdictions. Extend as needed.

    Args:
        jurisdiction_id: Internal jurisdiction ID

    Returns:
        MonQcle-format jurisdiction name
    """
    # Mapping of jurisdiction IDs to MonQcle names
    mapping = {
        "CA-LosAngeles": "Los Angeles, Los Angeles County, California, United States",
        # Add more mappings as jurisdictions are added
    }

    if jurisdiction_id not in mapping:
        raise ValueError(
            f"Unknown jurisdiction ID: {jurisdiction_id}. "
            f"Known jurisdictions: {list(mapping.keys())}"
        )

    return mapping[jurisdiction_id]
