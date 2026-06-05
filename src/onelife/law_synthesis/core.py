from loguru import logger
from pydantic import BaseModel


import re


class LawInduction(BaseModel):
    key_changes: str
    natural_language_law: str
    law_code: str

    @classmethod
    def parse(cls, response: str) -> "LawInduction":
        """
        Parse a response string containing XML-style tags to extract the three required fields.

        Args:
            response: String containing <keyChanges>, <naturalLanguageLaw>, and <lawCode> sections

        Returns:
            LawInduction: A new LawInduction instance with the parsed fields

        Raises:
            ValueError: If any required section is missing or malformed
        """

        key_changes_tag = "keyChanges"
        natural_language_law_tag = "naturalLanguageLaw"
        law_code_tag = "lawCode"

        # Define the patterns to match each section
        key_changes_pattern = rf"<{key_changes_tag}>(.*?)</{key_changes_tag}>"
        natural_language_law_pattern = (
            rf"<{natural_language_law_tag}>(.*?)</{natural_language_law_tag}>"
        )
        law_code_pattern = rf"<{law_code_tag}>(.*?)</{law_code_tag}>"

        # Extract each section using regex with re.DOTALL to match across lines
        key_changes_match = re.search(key_changes_pattern, response, re.DOTALL)
        natural_language_law_match = re.search(
            natural_language_law_pattern, response, re.DOTALL
        )
        law_code_match = re.search(law_code_pattern, response, re.DOTALL)

        # Check if all sections were found
        if not key_changes_match:
            raise ValueError("Missing <keyChanges> section in response")
        if not natural_language_law_match:
            raise ValueError("Missing <naturalLanguageLaw> section in response")
        if not law_code_match:
            raise ValueError("Missing <lawCode> section in response")

        # Extract and clean the content (strip whitespace)
        key_changes = key_changes_match.group(1).strip()
        natural_language_law = natural_language_law_match.group(1).strip()
        law_code = law_code_match.group(1).strip()

        # Remove the Python code block markers if present
        if law_code.startswith("```python"):
            law_code = law_code[9:]
        if law_code.endswith("```"):
            law_code = law_code[:-3]
        law_code = law_code.strip()

        return cls(
            key_changes=key_changes,
            natural_language_law=natural_language_law,
            law_code=law_code,
        )

    @classmethod
    def parse_multiple(cls, response: str) -> list["LawInduction"]:
        """
        Parse a response string that may contain multiple laws, each with their own XML-style tags.

        Args:
            response: String containing one or more sets of <keyChanges>, <naturalLanguageLaw>, and <lawCode> sections

        Returns:
            list[LawInduction]: A list of LawInduction instances parsed from the response

        Raises:
            ValueError: If any required section is missing or malformed
        """
        laws = []

        print(response)

        # Split the response by looking for the start of new law sections
        # We'll look for patterns that indicate the start of a new law
        # This is a simple approach - we split on <keyChanges> tags
        law_sections = re.split(r"<keyChanges>", response)

        # Remove any empty sections and add back the <keyChanges> tag to non-first sections
        for i, section in enumerate(law_sections):
            if not section.strip():
                continue

            # Add back the <keyChanges> tag to all sections except the first
            if i > 0:
                section = "<keyChanges>" + section

            try:
                law = cls.parse(section)
                laws.append(law)
            except ValueError as e:
                # Log the error but continue parsing other laws
                logger.warning(f"Failed to parse law section: {e}")
                continue

        if not laws:
            raise ValueError("No valid laws found in response")

        return laws
