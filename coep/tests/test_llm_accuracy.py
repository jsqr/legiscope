"""Regression tests for benchmark accuracy analysis question typing."""

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
	candidate_str = str(candidate)
	if candidate_str not in sys.path:
		sys.path.insert(0, candidate_str)

_MODULE_PATH = PROJECT_ROOT / "coep" / "analysis" / "LLM_accuracy.py"
_SPEC = importlib.util.spec_from_file_location("test_llm_accuracy_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
llm_accuracy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = llm_accuracy
_SPEC.loader.exec_module(llm_accuracy)


def _row(*, response_options: str, evaluation_mode: str = "whole_answer", query_text: str = "") -> dict[str, object]:
	return {
		"evaluation_mode": evaluation_mode,
		"query": query_text,
		"query_metadata": json.dumps(
			{
				"query_text": query_text,
				"response_options": response_options,
			}
		),
	}


class TestDeriveQuestionType:
	def test_classifies_binary_yes_no_questions(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options="Responses: Yes OR No",
				query_text="Does the jurisdiction have a law?",
			)
		)

		assert question_type == "Binary"

	def test_classifies_scalar_date_questions(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options="Responses: <enactment date> OR Unknown",
				query_text="On which date was the law enacted?",
			)
		)

		assert question_type == "Date"

	def test_classifies_single_option_current_through_date_questions(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options="Responses: <current-through date>",
				query_text="What is the current-through date of the ordinance?",
			)
		)

		assert question_type == "Date"

	def test_classifies_ssp_current_imp_as_categorical(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options=(
					"Responses: Known, current through date published in ordinance OR "
					"Known, partial current through date published in ordinance (month or day imputed) OR "
					"Unknown, reflects date of data collection"
				),
				query_text="Is the current-through date known or imputed?",
			)
		)

		assert question_type == "Categorical"

	def test_classifies_ssp_restrict_as_multi_select(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options=(
					"Responses: Cap on total number of programs or sites AND/OR "
					"Restrictions on mobile sites AND/OR No restrictions listed"
				),
				evaluation_mode="response_option",
				query_text="Does the ordinance require any restrictions on SSPs?",
			)
		)

		assert question_type == "Multi-select"

	def test_classifies_citation_or_unknown_as_categorical(self):
		question_type = llm_accuracy.derive_question_type(
			_row(
				response_options="Responses: <citation> OR Unknown",
				query_text="If yes, what is the citation of the relevant law?",
			)
		)

		assert question_type == "Categorical"