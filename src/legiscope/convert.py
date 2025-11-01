"""
Code to convert text files with outline structure and section headings to Markdown.
"""

from pydantic import BaseModel
from legiscope.utils import ask


class BooleanResult(BaseModel):
    """True/false result, or None, with explanation of reasoning."""

    answer: bool | None
    explanation: str
