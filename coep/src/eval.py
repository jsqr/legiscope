"""
Evaluation module for assessing the quality of legal query responses.
This module implements LLM-as-a-judge patterns to score generated answers
against ground truth human-authored answers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Literal, TypeAlias

import polars as pl
from loguru import logger
from polars import DataFrame
from pydantic import BaseModel, Field, model_validator

from legiscope.utils import LLMConfig


EvaluationErrorType: TypeAlias = Literal[
    "none",
    "abstention",
    "retrieval_failure",
    "retrieval_noise",
    "llm_failure",
    "reasoning_error",
    "hallucination",
    "output_contract_error",
    "other",
]


_SERIES_TITLE_BY_MONQCLE_REPORT_NAME = {
    "Drug_Paraphernalia_Laws_Standard_Report_20260501.csv": "DPL_2025_Consolidated",
    "SSP_Laws_Standard_Report_20260513.csv": "SSP_2025_Consolidated",
}


def expected_series_title_for_monqcle_report(
    monqcle_path: str | Path,
) -> str | None:
    """Return the project-specific series title for a known MonQcle report."""
    return _SERIES_TITLE_BY_MONQCLE_REPORT_NAME.get(Path(monqcle_path).name)


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
    error_type: EvaluationErrorType = Field(
        ...,
        description=(
            "Primary failure category for imperfect answers. Use `none` for correct answers, "
            "or one of: abstention, retrieval_failure, retrieval_noise, llm_failure, "
            "reasoning_error, hallucination, output_contract_error, other."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_like_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        expected_keys = {"score", "reasoning", "accuracy_label", "error_type"}
        if expected_keys.intersection(data):
            return data

        properties = data.get("properties")
        if isinstance(properties, dict) and expected_keys.issubset(properties):
            return properties

        return data

    model_config = {"extra": "ignore"}


class Evaluator:
    """Handles the evaluation of generated responses against ground truth."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        *,
        max_concurrency: int = 1,
    ):
        """
        Initialize the evaluator.

        Args:
            llm_config: Configuration for the judge LLM. If None, uses the powerful client.
            max_concurrency: Maximum number of concurrent evaluation requests.
        """
        self._thread_local = local()
        self._client_factory = None

        if llm_config is None:
            from legiscope.llm_config import Config

            # We want a powerful model for evaluation (Judge)
            # Config.get_powerful_client() already returns an Instructor client
            if Config.uses_self_hosted_llm():
                self._client_factory = Config.get_powerful_client
                self.client = self._client_factory()
            else:
                self.client = Config.get_powerful_client()
            self._request_params = Config.get_llm_params()
        else:
            # llm_config.client is already an Instructor client
            self.client = llm_config.client
            self._request_params = {
                "temperature": llm_config.temperature,
                "max_retries": llm_config.max_retries,
            }
            if llm_config.source == "self_hosted":
                self._client_factory = llm_config.client_factory

        self.max_concurrency = self._normalize_max_concurrency(max_concurrency)
        if self.max_concurrency > 1 and self._client_factory is None:
            logger.warning(
                "Evaluator max_concurrency > 1 requested without a self-hosted client factory; "
                "falling back to sequential evaluation for reliability."
            )
            self.max_concurrency = 1

    @staticmethod
    def _normalize_max_concurrency(value: int) -> int:
        """Return a safe, positive worker count."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid evaluator max_concurrency={value!r}; using sequential evaluation."
            )
            return 1

        if parsed < 1:
            logger.warning(
                f"Non-positive evaluator max_concurrency={parsed}; using sequential evaluation."
            )
            return 1

        return parsed

    def _get_client(self):
        """Return a thread-local client when concurrent evaluation is enabled."""
        if self.max_concurrency == 1 or self._client_factory is None:
            return self.client

        thread_client = getattr(self._thread_local, "client", None)
        if thread_client is None:
            thread_client = self._client_factory()
            self._thread_local.client = thread_client
        return thread_client

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
        4. Failure mode: If the answer is not correct, what is the primary error type?

        Assign a score from 0 to 10, where:
        - 10: Perfect match in facts and nuance.
        - 7-9: Mostly correct, minor details missing or slightly different phrasing.
        - 4-6: Partially correct, misses key points or has minor inaccuracies.
        - 0-3: Completely wrong, irrelevant, or dangerous hallucination.

        Important notes:
        - The ground truth may contain binary results, where 0 = "No" and 1 = "Yes". 
          The generated answer may use different phrasing (e.g., "No", "False", "0", "Negative" for 0). 
          Focus on the meaning rather than exact wording.
                - When the generated payload includes both a short answer and supporting reasoning, treat the short answer as authoritative for correctness.
                - If the short answer and reasoning conflict, prioritize the short answer's semantic match to the ground truth for `accuracy_label` and `error_type`.
                    Do not mark an answer incorrect solely because the reasoning text is inconsistent when the short answer itself is correct.

                OUTPUT REQUIREMENTS:
                - Return exactly one JSON object representing an EvaluationResult instance.
                                - Use these exact top-level keys: `score`, `reasoning`, `accuracy_label`, `error_type`.
                - `score` must be an integer from 0 to 10.
                - `accuracy_label` must be one of: `Correct`, `Partially Correct`, `Incorrect`.
                                - `error_type` must be one of:
                                    - `none` for correct answers with no material failure
                                    - `abstention` when the system declines to answer or says it cannot answer
                                    - `retrieval_failure` when the answer appears limited by missing or filtered-out evidence
                                    - `retrieval_noise` when retrieved legal text is present but the answer is driven by off-target, noisy, or non-controlling context
                                    - `llm_failure` when the model times out, errors, or returns an unusable payload rather than a substantive answer
                                    - `reasoning_error` when evidence exists but the answer extracts or combines it incorrectly
                                    - `hallucination` when the answer invents unsupported facts
                                    - `output_contract_error` when the answer violates the required response format
                                    - `other` for any remaining failure mode
                - Do not return JSON Schema or metadata.
                - Never include keys like `description`, `properties`, `title`, `type`, `required`, or `$defs`.
                - Do not wrap the answer inside `properties` or any other container.
                - No commentary, no Markdown fences, no prose outside the JSON object.

                OUTPUT TEMPLATE:
                {{
                    "score": 0,
                    "reasoning": "...",
                    "accuracy_label": "Incorrect",
                    "error_type": "other"
                }}
        """

        try:
            client = self._get_client()
            return client.chat.completions.create(
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
                error_type="other",
            )

    def evaluate_batch(
        self, df: DataFrame, question_col: str, answer_col: str, truth_col: str
    ) -> DataFrame:
        """
        Run evaluation on a DataFrame containing results and ground truth.
        """
        rows = list(df.iter_rows(named=True))
        record_count = len(rows)
        scores = []
        reasons = []
        labels = []
        error_types = []

        worker_count = min(self.max_concurrency, record_count) if record_count else 1
        logger.info(
            f"Starting evaluation of {record_count} records with max_concurrency={worker_count}..."
        )

        results: list[EvaluationResult] = []
        if record_count == 0:
            results = []
        elif worker_count == 1:
            results = [
                self.evaluate_response(
                    question=row[question_col],
                    generated_answer=row[answer_col],
                    ground_truth=row[truth_col],
                )
                for row in rows
            ]
        else:
            results = [
                EvaluationResult(
                    score=0,
                    reasoning="Evaluation execution failed before a result was recorded.",
                    accuracy_label="Incorrect",
                    error_type="other",
                )
                for _ in rows
            ]
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="benchmark-eval",
            ) as executor:
                futures = {
                    executor.submit(
                        self.evaluate_response,
                        question=row[question_col],
                        generated_answer=row[answer_col],
                        ground_truth=row[truth_col],
                    ): index
                    for index, row in enumerate(rows)
                }

                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        logger.error(f"Concurrent evaluation worker failed: {exc}")
                        results[index] = EvaluationResult(
                            score=0,
                            reasoning=f"Evaluation execution failed: {exc}",
                            accuracy_label="Incorrect",
                            error_type="other",
                        )

        for result in results:
            scores.append(result.score)
            reasons.append(result.reasoning)
            labels.append(result.accuracy_label)
            error_types.append(result.error_type)

        # Add evaluation columns to the dataframe
        return df.with_columns(
            [
                pl.Series("eval_score", scores),
                pl.Series("eval_reason", reasons),
                pl.Series("eval_label", labels),
                pl.Series("eval_error_type", error_types),
            ]
        )


def load_and_filter_monqcle(
    monqcle_path: str,
    jurisdiction_name: str,
    series_title: str | None = None,
) -> pl.DataFrame:
    """
    Load MonQcle Standard Report and filter to target jurisdiction.

    The MonQcle data is in wide format (one row per jurisdiction, variables as columns).
    This function filters to the target jurisdiction and series.

    Args:
        monqcle_path: Path to MonQcle Standard Report CSV
        jurisdiction_name: Full jurisdiction name (e.g., "Philadelphia, Philadelphia County, Pennsylvania, United States")
        series_title: Series to filter on. When omitted, known COEP reports use
            a hard-coded project-specific series title based on the report name.
            If the requested series is absent for this jurisdiction, a
            report-local single available series is used as a fallback.

    Returns:
        DataFrame with single row for target jurisdiction
    """
    df = pl.read_csv(monqcle_path)
    resolved_series_title = series_title or expected_series_title_for_monqcle_report(
        monqcle_path
    )

    jurisdiction_rows = df.filter(pl.col("name") == jurisdiction_name)
    if len(jurisdiction_rows) == 0:
        available = df["name"].unique().sort().to_list()
        raise ValueError(
            f"No records found for jurisdiction '{jurisdiction_name}'. "
            f"Available jurisdictions: {available[:20]}"
        )

    filtered = jurisdiction_rows
    if resolved_series_title:
        filtered = jurisdiction_rows.filter(
            pl.col("series_title") == resolved_series_title
        )
        if len(filtered) == 0:
            available_series = (
                jurisdiction_rows["series_title"].drop_nulls().unique().sort().to_list()
            )
            if len(available_series) == 1:
                inferred_series = str(available_series[0])
                logger.info(
                    f"Series '{resolved_series_title}' not found in {monqcle_path} for "
                    f"{jurisdiction_name}; using only available series '{inferred_series}'"
                )
                filtered = jurisdiction_rows.filter(
                    pl.col("series_title") == inferred_series
                )
            else:
                raise ValueError(
                    f"No records found for jurisdiction '{jurisdiction_name}' with series '{resolved_series_title}'. "
                    f"Available series for this jurisdiction: {available_series}"
                )

    if len(filtered) > 1:
        if "through_to" in filtered.columns:
            filtered = (
                filtered.with_columns(
                    pl.col("through_to")
                    .cast(pl.String)
                    .str.strptime(pl.Date, strict=False)
                    .alias("_through_to_date")
                )
                .sort(
                    by=["_through_to_date", "through_to"],
                    descending=[True, True],
                    nulls_last=True,
                )
                .drop("_through_to_date")
            )
            logger.warning(
                "Multiple MonQcle rows matched %s; using the row with the most recent through_to date (%s)"
                % (jurisdiction_name, filtered["through_to"][0])
            )
        else:
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
    This melts it to long format with columns: variable_name, ground_truth,
    ground_truth_citation

    Args:
        monqcle_row: Single-row DataFrame from MonQcle
        variable_names: List of variable names to extract (from queries file)

    Returns:
        DataFrame with variable_name, ground_truth, and ground_truth_citation columns
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
        citation_value = row_dict.get(f"_citations_{var_name}", None)
        # Convert MonQcle's "-" placeholder to None/empty
        if value == "-" or value is None:
            ground_truth = ""
        else:
            ground_truth = str(value)
        if citation_value == "-" or citation_value is None:
            ground_truth_citation = ""
        else:
            ground_truth_citation = str(citation_value)
        records.append(
            {
                "variable_name": var_name,
                "ground_truth": ground_truth,
                "ground_truth_citation": ground_truth_citation,
            }
        )

    result = pl.DataFrame(records)
    logger.info(f"Melted {len(result)} variables to long format")
    return result


# Mapping of legacy combined query variables to their MonQcle source columns.
# New nested COEP query CSVs should target the split variables directly; these
# compatibility aliases remain for older result files and benchmark queries.
_LEGACY_COMBINED_VARIABLE_SOURCES: dict[str, list[tuple[str, str]]] = {
    "dp_collected_combined": [
        ("dp_collected", "Collected"),
        ("dp_valid_imp", "Valid/Imp"),
    ],
    "dp_state_fed_combined": [
        ("dp_state_fed_reference", "References state/federal law"),
        ("dp_state_fed_citation", "Citation"),
    ],
}


def _dedupe_variable_names(variable_names: list[str]) -> list[str]:
    """Return requested variable names with original order preserved."""
    seen: set[str] = set()
    deduped: list[str] = []
    for variable_name in variable_names:
        stripped = str(variable_name).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped


def expand_combined_variables(
    monqcle_row: pl.DataFrame, variable_names: list[str]
) -> pl.DataFrame:
    """Add synthetic columns to a MonQcle row for legacy combined variables.

    Some query variables (e.g. ``dp_collected_combined``) combine multiple
    MonQcle columns.  This function detects those variables in
    *variable_names*, builds a combined ground-truth string from the
    source columns, and adds it as a new column on *monqcle_row*.

    Args:
        monqcle_row: Single-row MonQcle DataFrame.
        variable_names: Variable names requested by the queries.

    Returns:
        The MonQcle row with any necessary combined columns added.
    """
    combined_vars = [
        v for v in variable_names if v in _LEGACY_COMBINED_VARIABLE_SOURCES
    ]
    if not combined_vars:
        return monqcle_row

    logger.info(f"Expanding {len(combined_vars)} combined variable(s): {combined_vars}")
    row_dict = monqcle_row.to_dicts()[0]
    new_cols: list[pl.Expr] = []

    for var_name in combined_vars:
        parts: list[str] = []
        citation_parts: list[str] = []
        for col, label in _LEGACY_COMBINED_VARIABLE_SOURCES[var_name]:
            val = row_dict.get(col)
            val_str = str(val) if val not in [None, "-"] else ""
            parts.append(f"{label}: {val_str}")
            citation_col = f"_citations_{col}"
            citation_val = row_dict.get(citation_col)
            citation_str = str(citation_val) if citation_val not in [None, "-"] else ""
            if citation_str:
                citation_parts.append(f"{label}: {citation_str}")
        combined_truth = "\n".join(parts).strip()
        combined_citation = "\n".join(citation_parts).strip()
        new_cols.append(pl.lit(combined_truth).alias(var_name))
        new_cols.append(pl.lit(combined_citation).alias(f"_citations_{var_name}"))

    monqcle_row = monqcle_row.with_columns(new_cols)
    logger.info(f"Added combined columns to MonQcle data: {combined_vars}")
    return monqcle_row


def prepare_ground_truth_for_variables(
    monqcle_row: pl.DataFrame,
    variable_names: list[str],
) -> pl.DataFrame:
    """Build long-form ground truth with split variables primary and legacy compatibility optional."""
    requested_variables = _dedupe_variable_names(variable_names)
    compatibility_variables = [
        variable_name
        for variable_name in requested_variables
        if variable_name in _LEGACY_COMBINED_VARIABLE_SOURCES
    ]

    if compatibility_variables:
        logger.info(
            "Using legacy combined-variable compatibility for: "
            f"{compatibility_variables}"
        )
        monqcle_row = expand_combined_variables(monqcle_row, compatibility_variables)

    return melt_monqcle_to_long(monqcle_row, requested_variables)


def prioritize_ground_truth_matches(results_df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows with matched ground truth first while preserving query order within groups."""
    required_columns = {"ground_truth_available", "benchmark_row_id"}
    missing_columns = required_columns.difference(results_df.columns)
    if missing_columns:
        raise ValueError(
            "Results dataframe is missing required ordering columns: "
            + ", ".join(sorted(missing_columns))
        )

    sort_columns = ["ground_truth_available", "benchmark_row_id"]
    descending = [True, False]
    if "evaluation_subquestion_index" in results_df.columns:
        sort_columns.append("evaluation_subquestion_index")
        descending.append(False)

    return results_df.sort(
        by=sort_columns,
        descending=descending,
        nulls_last=True,
    )


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
        "AZ-Tucson": "Tucson, Pima County, Arizona, United States",
        "AZ-Yuma": "Yuma, Yuma County, Arizona, United States",
        "CA-Alhambra": "Alhambra, Los Angeles County, California, United States",
        "CA-Anaheim": "Anaheim, Orange County, California, United States",
        "CA-Antioch": "Antioch, Contra Costa County, California, United States",
        "CA-BaldwinPark": "Baldwin Park, Los Angeles County, California, 91706, United States",
        "CA-Chico": "Chico, Butte County, California, United States",
        "CA-Corona": "Corona, Riverside County, California, United States",
        "CA-Fullerton": "Fullerton, Orange County, California, United States",
        "CA-LosAngeles": "Los Angeles, Los Angeles County, California, United States",
        "CA-Menifee": "Menifee, Riverside County, California, United States",
        "CA-Murrieta": "Murrieta, Riverside County, California, United States",
        "CA-Ontario": "Ontario, San Bernardino County, California, United States",
        "CA-Oxnard": "Oxnard, Ventura County, California, United States",
        "CA-Sacramento": "Sacramento, Sacramento County, California, United States",
        "CA-SanBernardino": "San Bernardino, San Bernardino County, California, United States",
        "CA-ThousandOaks": "Thousand Oaks, Ventura County, California, United States",
        "CA-Visalia": "Visalia, Tulare County, California, United States",
        "CO-ColoradoSprings": "Colorado Springs, El Paso County, Colorado, United States",
        "CT-Waterbury": "Waterbury, Naugatuck Valley Planning Region, Connecticut, United States",
        "FL-Hollywood": "Hollywood, Broward County, Florida, United States",
        "FL-Palmbay": "Palm Bay, Brevard County, Florida, United States",
        "FL-PembrokePines": "Pembroke Pines, Broward County, Florida, United States",
        "FL-PompanoBeach": "Pompano Beach, Broward County, Florida, United States",
        "HI-Honolulu": "Honolulu, Honolulu County, Hawaii, United States",
        "IA-IowaCity": "Iowa City, Johnson County, Iowa, United States",
        "ID-Boise": "Boise, Ada County, Idaho, United States",
        "IN-Carmel": "Carmel, Hamilton County, Indiana, United States",
        "IN-Fishers": "Fishers, Hamilton County, Indiana, United States",
        "IN-FortWayne": "Fort Wayne, Allen County, Indiana, United States",
        "IN-Hammond": "Hammond, Lake County, Indiana, United States",
        "KY-LexingtonFayetteCounty": "Lexington, Fayette County, Kentucky, United States",
        "MI-Dearborn": "Dearborn, Wayne County, Michigan, United States",
        "MI-SterlingHeights": "Sterling Heights, Macomb County, Michigan, United States",
        "MN-BrooklynPark": "Brooklyn Park, Hennepin County, Minnesota, United States",
        "NC-Cary": "Cary, Wake County, North Carolina, United States",
        "NC-Greenville": "Greenville, Pitt County, North Carolina, United States",
        "NH-Manchester": "Manchester, Hillsborough County, New Hampshire, United States",
        "NM-Albuquerque": "Albuquerque, Bernalillo County, New Mexico, United States",
        "OH-Cleveland": "Cleveland, Cuyahoga County, Ohio, United States",
        "OH-Parma": "Parma, Cuyahoga County, Ohio, United States",
        "OH-Toledo": "Toledo, Lucas County, Ohio, United States",
        "PA-Philadelphia": "Philadelphia, Philadelphia County, Pennsylvania, United States",
        "SD-SiouxFalls": "Sioux Falls, Minnehaha County, South Dakota, United States",
        "TX-Dallas": "Dallas, Dallas County, Texas, United States",
        "TX-FortWorth": "Fort Worth, Tarrant County, Texas, United States",
        "TX-Longview": "Longview, Gregg County, Texas, United States",
        "TX-Tyler": "Tyler, Smith County, Texas, United States",
        "UT-SaltLakeCity": "Salt Lake City, Salt Lake County, Utah, United States",
        "UT-WestJordan": "West Jordan, Salt Lake County, Utah, United States",
    }

    if jurisdiction_id not in mapping:
        raise ValueError(
            f"Unknown jurisdiction ID: {jurisdiction_id}. "
            f"Known jurisdictions: {list(mapping.keys())}"
        )

    return mapping[jurisdiction_id]
