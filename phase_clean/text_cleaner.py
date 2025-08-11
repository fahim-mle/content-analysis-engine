# phase_clean/text_cleaner.py
"""Core text cleaning functionality for research papers."""

import re
import unicodedata
from typing import Any, Dict, List, Tuple

from .utils import get_logger

logger = get_logger(__name__)


class ResearchPaperCleaner:
    """Clean and normalize research paper text extracted from PDFs."""

    def __init__(self):
        # Common patterns to remove or fix
        self.patterns = {
            # Page numbers (standalone digits)
            "page_numbers": re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),
            # ArXiv identifiers
            "arxiv_ids": re.compile(
                r"arXiv:\d{4}\.\d{4,5}v\d+(?:\s*\[[\w\.-]+\])?(?:\s*\d+\s+\w+\s+\d{4})?",
                re.IGNORECASE,
            ),
            # Excessive whitespace (3+ spaces, tabs, newlines)
            "excess_whitespace": re.compile(r"\s{3,}"),
            # Multiple consecutive newlines
            "multiple_newlines": re.compile(r"\n\s*\n\s*\n+"),
            # Hyphenated words across lines (word- \nword)
            "hyphenation": re.compile(r"(\w+)-\s*\n\s*(\w+)"),
            # Common PDF artifacts
            "pdf_artifacts": re.compile(
                r"[^\w\s\.\,\!\?\:\;\"\'\(\)\[\]\{\}\-\+\=\<\>\@\#\$\%\&\*\/\\\|\~\`\^]"
            ),
            # Figure/table captions (often noise)
            "figure_table": re.compile(
                r"(?:Figure|Table|Fig\.|Tab\.)\s*\d+[:\.]?\s*[^\n]*", re.IGNORECASE
            ),
            # Email addresses (usually author info, can be noise)
            "emails": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            # URLs
            "urls": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            # Common LaTeX commands
            "latex_commands": re.compile(
                r"\\[a-zA-Z]+\*?\s*(?:\s*\[[^\]]*\])?\s*(?:\s*\{[^}]*\})*"
            ),
            # Mathematical expressions (basic)
            "math_expressions": re.compile(r"\$[^$]*\$\|\$\$[^$]*\$\$"),
        }

        # Academic section headers to preserve
        self.section_headers = [
            "abstract",
            "introduction",
            "related work",
            "methodology",
            "method",
            "approach",
            "experiments",
            "results",
            "evaluation",
            "discussion",
            "conclusion",
            "conclusions",
            "references",
            "bibliography",
            "acknowledgments",
            "acknowledgements",
            "appendix",
        ]

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters and fix encoding issues."""
        try:
            # Normalize to composed form
            text = unicodedata.normalize("NFKC", text)

            # Fix common encoding issues
            replacements = {
                "´": "'",
                "`": "'",
                "’": "'", "‘": "'",
                '”': '"',
                '“': '"',
                "–": "-",
                "—": "-",
                "…": "...",
                "¨": "",
                "¯": "-",
                "ﬁ": "fi",
                "ﬂ": "fl",
                "ﬀ": "ff",
                "ﬃ": "ffi",
                "ﬄ": "ffl",
                "α": "alpha",
                "β": "beta",
                "γ": "gamma",
                "δ": "delta",
                "ε": "epsilon",
                "θ": "theta",
                "λ": "lambda",
                "μ": "mu",
                "π": "pi",
                "σ": "sigma",
                "τ": "tau",
                "φ": "phi",
                "≤": "<= ",
                "≥": ">= ",
                "≠": "!= ",
                "≈": "~= ",
                "∞": "infinity",
            }

            for old, new in replacements.items():
                text = text.replace(old, new)

        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")

        return text

    def remove_artifacts(self, text: str) -> str:
        """Remove common PDF extraction artifacts."""
        original_length = len(text)

        # Remove ArXiv identifiers
        text = self.patterns["arxiv_ids"].sub("", text)

        # Remove page numbers (but be careful not to remove legitimate numbers)
        text = self.patterns["page_numbers"].sub("", text)

        # Remove URLs and emails (usually metadata)
        text = self.patterns["urls"].sub("[URL]", text)
        text = self.patterns["emails"].sub("[EMAIL]", text)

        # Remove basic LaTeX commands
        text = self.patterns["latex_commands"].sub(" ", text)

        # Remove mathematical expressions (preserve space)
        text = self.patterns["math_expressions"].sub(" [MATH] ", text)

        # Clean up PDF artifacts (non-printable chars)
        text = self.patterns["pdf_artifacts"].sub(" ", text)

        cleaned_length = len(text)
        logger.debug(f"Removed artifacts: {original_length} -> {cleaned_length} chars")

        return text

    def fix_hyphenation(self, text: str) -> str:
        """Fix words broken by hyphenation across lines."""

        def hyphen_replacer(match):
            word1, word2 = match.groups()
            # Only join if both parts are alphabetic
            if word1.isalpha() and word2.isalpha():
                return word1 + word2
            else:
                return match.group(0)

        return self.patterns["hyphenation"].sub(hyphen_replacer, text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Fix hyphenated words first
        text = self.fix_hyphenation(text)

        # Replace multiple consecutive newlines with double newline
        text = self.patterns["multiple_newlines"].sub("\n\n", text)

        # Replace excessive whitespace with single space
        text = self.patterns["excess_whitespace"].sub(" ", text)

        # Clean up line endings
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)
            elif lines and lines[-1]:  # Preserve paragraph breaks
                lines.append("")

        return "\n".join(lines)

    def extract_and_clean_sections(self, text: str) -> Dict[str, str]:
        """Extract and identify paper sections."""
        sections = {}
        current_section = "body"
        current_content = []

        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line is a section header
            found_section = None
            for section in self.section_headers:
                if (
                    line_lower == section
                    or line_lower.startswith(section + " ")
                    or line_lower.endswith(" " + section)
                ):
                    found_section = section
                    break

            if found_section:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = found_section
                current_content = []
            else:
                # Add to current section
                if line.strip():
                    current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def assess_quality(self, text: str) -> Dict[str, Any]:
        """Assess text quality and detect potential issues."""
        quality = {
            "length": len(text),
            "word_count": len(text.split()),
            "issues": [],
            "score": 1.0,
        }

        # Too short
        if quality["length"] < 500:
            quality["issues"].append("too_short")
            quality["score"] *= 0.3

        # Too many non-alphabetic characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.6:
            quality["issues"].append("low_alpha_ratio")
            quality["score"] *= 0.5

        # Check for gibberish (too many single-letter words)
        words = text.split()
        single_letters = sum(1 for w in words if len(w) == 1)
        if single_letters / max(len(words), 1) > 0.1:
            quality["issues"].append("potential_gibberish")
            quality["score"] *= 0.4

        # Check for excessive repetition
        if len(set(words)) / max(len(words), 1) < 0.3:
            quality["issues"].append("excessive_repetition")
            quality["score"] *= 0.6

        return quality

    def clean_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Complete text cleaning pipeline.

        Args:
            text: Raw text to clean

        Returns:
            Tuple of (cleaned_text, quality_metrics)
        """
        if not text or not text.strip():
            return "", {
                "length": 0,
                "word_count": 0,
                "issues": ["empty_text"],
                "score": 0.0,
            }

        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)

        # Step 2: Remove artifacts
        text = self.remove_artifacts(text)

        # Step 3: Normalize whitespace
        text = self.normalize_whitespace(text)

        # Step 4: Final cleanup
        text = text.strip()

        # Step 5: Quality assessment
        quality = self.assess_quality(text)

        return text, quality


def process_paper_text(
    paper_data: Dict[str, Any], cleaner: ResearchPaperCleaner = None
) -> Dict[str, Any]:
    """
    Process and clean text for a single paper.

    Args:
        paper_data: Paper metadata with full_text field
        cleaner: Optional cleaner instance (creates new if None)

    Returns:
        Updated paper data with cleaned_text and quality metrics
    """
    if cleaner is None:
        cleaner = ResearchPaperCleaner()

    arxiv_id = paper_data.get("arxiv_id", "unknown")

    if "full_text" not in paper_data or not paper_data["full_text"]:
        logger.warning(f"No full_text found for paper {arxiv_id}")
        paper_data["cleaned_text"] = ""
        paper_data["text_quality"] = {
            "length": 0,
            "word_count": 0,
            "issues": ["no_text"],
            "score": 0.0,
        }
        return paper_data

    try:
        cleaned_text, quality = cleaner.clean_text(paper_data["full_text"])

        paper_data["cleaned_text"] = cleaned_text
        paper_data["text_quality"] = quality
        paper_data["text_sections"] = cleaner.extract_and_clean_sections(cleaned_text)

        logger.info(
            f"Cleaned {arxiv_id}: {quality['length']} chars, quality score: {quality['score']:.2f}"
        )

        if quality["issues"]:
            logger.debug(f"Issues in {arxiv_id}: {', '.join(quality['issues'])}")

    except Exception as e:
        logger.error(f"Failed to clean text for {arxiv_id}: {e}")
        paper_data["cleaned_text"] = paper_data["full_text"]  # Fallback to original
        paper_data["text_quality"] = {
            "length": 0,
            "word_count": 0,
            "issues": ["processing_error"],
            "score": 0.1,
        }
        paper_data["text_sections"] = {"body": paper_data["full_text"]}

    return paper_data