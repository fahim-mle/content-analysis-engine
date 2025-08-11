# phase_clean/section_parser.py
"""Advanced section parsing and structure detection for research papers."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Section:
    """Represents a section in a research paper."""

    name: str
    content: str
    start_pos: int
    end_pos: int
    level: int  # Header level (1=main section, 2=subsection, etc.)
    confidence: float  # How confident we are this is a real section


class ResearchPaperSectionParser:
    """Parse and identify sections in research papers."""

    def __init__(self):
        # Main section patterns (ordered by typical paper structure)
        self.main_sections = {
            "abstract": {
                "patterns": [
                    r"^abstract\s*$",
                    r"^abstract\s*[:\-]",
                    r"^\d+\.?\s*abstract\s*$",
                ],
                "aliases": ["summary"],
                "expected_position": 0.0,  # Expected relative position in paper
                "min_words": 50,
                "max_words": 500,
            },
            "introduction": {
                "patterns": [
                    r"^introduction\s*$",
                    r"^1\.?\s*introduction\s*$",
                    r"^\d+\.?\s*introduction\s*$",
                ],
                "aliases": ["intro"],
                "expected_position": 0.1,
                "min_words": 100,
                "max_words": 2000,
            },
            "related_work": {
                "patterns": [
                    r"^related\s+works?\s*$",
                    r"^background\s*$",
                    r"^literature\s+review\s*$",
                    r"^\d+\.?\s*related\s+works?\s*$",
                    r"^\d+\.?\s*background\s*$",
                ],
                "aliases": ["background", "literature_review"],
                "expected_position": 0.2,
                "min_words": 100,
                "max_words": 2000,
            },
            "methodology": {
                "patterns": [
                    r"^methodology\s*$",
                    r"^methods?\s*$",
                    r"^approach\s*$",
                    r"^model\s*$",
                    r"^\d+\.?\s*methodology\s*$",
                    r"^\d+\.?\s*methods?\s*$",
                    r"^\d+\.?\s*approach\s*$",
                ],
                "aliases": ["method", "approach", "model", "framework"],
                "expected_position": 0.3,
                "min_words": 200,
                "max_words": 3000,
            },
            "experiments": {
                "patterns": [
                    r"^experiments?\s*$",
                    r"^experimental\s+setup\s*$",
                    r"^evaluation\s*$",
                    r"^\d+\.?\s*experiments?\s*$",
                    r"^\d+\.?\s*evaluation\s*$",
                ],
                "aliases": ["evaluation", "experimental_setup"],
                "expected_position": 0.5,
                "min_words": 200,
                "max_words": 3000,
            },
            "results": {
                "patterns": [
                    r"^results?\s*$",
                    r"^results?\s+and\s+discussion\s*$",
                    r"^\d+\.?\s*results?\s*$",
                    r"^\d+\.?\s*results?\s+and\s+discussion\s*$",
                ],
                "aliases": ["findings"],
                "expected_position": 0.6,
                "min_words": 100,
                "max_words": 2500,
            },
            "discussion": {
                "patterns": [
                    r"^discussion\s*$",
                    r"^analysis\s*$",
                    r"^\d+\.?\s*discussion\s*$",
                    r"^\d+\.?\s*analysis\s*$",
                ],
                "aliases": ["analysis"],
                "expected_position": 0.7,
                "min_words": 100,
                "max_words": 2000,
            },
            "conclusion": {
                "patterns": [
                    r"^conclusions?\s*$",
                    r"^concluding\s+remarks\s*$",
                    r"^\d+\.?\s*conclusions?\s*$",
                    r"^\d+\.?\s*concluding\s+remarks\s*$",
                ],
                "aliases": ["conclusions", "concluding_remarks"],
                "expected_position": 0.8,
                "min_words": 50,
                "max_words": 1000,
            },
            "references": {
                "patterns": [
                    r"^references?\s*$",
                    r"^bibliography\s*$",
                    r"^\d+\.?\s*references?\s*$",
                    r"^\d+\.?\s*bibliography\s*$",
                ],
                "aliases": ["bibliography"],
                "expected_position": 0.9,
                "min_words": 10,
                "max_words": 5000,
            },
            "acknowledgments": {
                "patterns": [
                    r"^acknowledgments?\s*$",
                    r"^acknowledgements?\s*$",
                    r"^\d+\.?\s*acknowledgments?\s*$",
                    r"^\d+\.?\s*acknowledgements?\s*$",
                ],
                "aliases": ["acknowledgements"],
                "expected_position": 0.85,
                "min_words": 10,
                "max_words": 500,
            },
            "appendix": {
                "patterns": [
                    r"^appendix\s*$",
                    r"^appendices\s*$",
                    r"^appendix\s+[a-z]\s*$",
                    r"^\d+\.?\s*appendix\s*$",
                ],
                "aliases": ["appendices"],
                "expected_position": 0.95,
                "min_words": 50,
                "max_words": 10000,
            },
        }

        # Compile all patterns
        self.compiled_patterns = {}
        for section_name, section_info in self.main_sections.items():
            self.compiled_patterns[section_name] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in section_info["patterns"]
            ]

    def detect_section_headers(self, text: str) -> List[Tuple[str, int, int, float]]:
        """
        Detect section headers in text.

        Returns:
            List of (section_name, start_pos, line_num, confidence) tuples
        """
        headers = []
        lines = text.split("\n")

        for line_num, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) > 200:  # Skip empty or very long lines
                continue

            # Calculate position in document
            char_pos = sum(len(lines[i]) + 1 for i in range(line_num))

            # Check against all section patterns
            for section_name, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.match(line_clean):
                        confidence = self._calculate_header_confidence(
                            line_clean, section_name, line_num, len(lines)
                        )
                        headers.append((section_name, char_pos, line_num, confidence))
                        break

        return headers

    def _calculate_header_confidence(
        self, line: str, section_name: str, line_num: int, total_lines: int
    ) -> float:
        """Calculate confidence score for a detected header."""
        confidence = 0.5  # Base confidence

        # Position-based confidence
        expected_pos = self.main_sections[section_name]["expected_position"]
        actual_pos = line_num / max(total_lines, 1)
        pos_diff = abs(expected_pos - actual_pos)

        if pos_diff < 0.1:
            confidence += 0.3
        elif pos_diff < 0.3:
            confidence += 0.1
        else:
            confidence -= 0.2

        # Line characteristics
        if line.isupper():
            confidence += 0.1

        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            confidence += 0.2

        if len(line) < 50:  # Headers are typically short
            confidence += 0.1

        # Avoid false positives
        if any(word in line.lower() for word in ["figure", "table", "eq.", "equation"]):
            confidence -= 0.4

        return max(0.0, min(1.0, confidence))

    def extract_sections(self, text: str) -> Dict[str, Section]:
        """
        Extract all sections from the text.

        Returns:
            Dictionary mapping section names to Section objects
        """
        if not text.strip():
            return {}

        headers = self.detect_section_headers(text)

        # Filter and sort headers by confidence and position
        headers = [
            (name, pos, line, conf) for name, pos, line, conf in headers if conf > 0.3
        ]
        headers.sort(
            key=lambda x: (x[1], -x[3])
        )  # Sort by position, then by confidence (desc)

        # Remove duplicate sections (keep highest confidence)
        seen_sections = set()
        unique_headers = []
        for name, pos, line, conf in headers:
            if name not in seen_sections:
                unique_headers.append((name, pos, line, conf))
                seen_sections.add(name)

        headers = unique_headers
        sections = {}

        # Extract content between headers
        for i, (section_name, start_pos, start_line, confidence) in enumerate(headers):
            # Determine end position
            if i + 1 < len(headers):
                end_pos = headers[i + 1][1]
            else:
                end_pos = len(text)

            # Extract content (skip the header line itself)
            lines = text[start_pos:end_pos].split("\n")
            if lines:
                content_lines = lines[1:]  # Skip header line
                content = "\n".join(content_lines).strip()
            else:
                content = ""

            # Validate section content
            word_count = len(content.split())
            section_config = self.main_sections.get(section_name, {})
            min_words = section_config.get("min_words", 0)
            max_words = section_config.get("max_words", 10000)

            if min_words <= word_count <= max_words:
                sections[section_name] = Section(
                    name=section_name,
                    content=content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    level=1,
                    confidence=confidence,
                )
            else:
                logger.debug(
                    f"Section {section_name} rejected: {word_count} words (expected {min_words}-{max_words})"
                )

        # Handle body text (text not assigned to any section)
        if not sections and text.strip():
            sections["body"] = Section(
                name="body",
                content=text.strip(),
                start_pos=0,
                end_pos=len(text),
                level=1,
                confidence=0.8,
            )
        elif headers:
            # Extract text before first section
            first_header_pos = min(pos for _, pos, _, _ in headers)
            if (
                first_header_pos > 100
            ):  # If there's substantial text before first section
                pre_content = text[:first_header_pos].strip()
                if len(pre_content.split()) > 50:
                    sections["introduction_unlabeled"] = Section(
                        name="introduction_unlabeled",
                        content=pre_content,
                        start_pos=0,
                        end_pos=first_header_pos,
                        level=1,
                        confidence=0.6,
                    )

        return sections

    def create_chunked_sections(
        self, sections: Dict[str, Section], chunk_size: int = 1000
    ) -> List[Dict]:
        """
        Create chunks from sections for ML processing.

        Args:
            sections: Dictionary of sections
            chunk_size: Target chunk size in characters

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_id = 0

        # Priority order for sections
        section_order = [
            "abstract",
            "introduction",
            "related_work",
            "methodology",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "acknowledgments",
            "references",
            "appendix",
            "body",
            "introduction_unlabeled",
        ]

        for section_name in section_order:
            if section_name not in sections:
                continue

            section = sections[section_name]
            content = section.content

            if len(content) <= chunk_size:
                # Section fits in one chunk
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "section": section_name,
                        "text": content,
                        "char_count": len(content),
                        "word_count": len(content.split()),
                        "section_confidence": section.confidence,
                    }
                )
                chunk_id += 1
            else:
                # Split large section into smaller chunks
                sentences = re.split(r"(?<=[.!?])\s+", content)
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)

                    if current_length + sentence_length > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "section": section_name,
                                "text": chunk_text,
                                "char_count": len(chunk_text),
                                "word_count": len(chunk_text.split()),
                                "section_confidence": section.confidence,
                            }
                        )
                        chunk_id += 1
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(sentence)
                    current_length += sentence_length + 1  # +1 for space

                # Save final chunk if any content remains
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "section": section_name,
                            "text": chunk_text,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                            "section_confidence": section.confidence,
                        }
                    )
                    chunk_id += 1

        return chunks

    def get_section_summary(self, sections: Dict[str, Section]) -> Dict:
        """Generate summary statistics for extracted sections."""
        if not sections:
            return {"total_sections": 0, "total_words": 0, "sections_found": []}

        total_words = sum(len(section.content.split()) for section in sections.values())
        sections_found = list(sections.keys())
        avg_confidence = sum(section.confidence for section in sections.values()) / len(
            sections
        )

        return {
            "total_sections": len(sections),
            "total_words": total_words,
            "sections_found": sections_found,
            "avg_confidence": avg_confidence,
            "has_abstract": "abstract" in sections,
            "has_introduction": "introduction" in sections
            or "introduction_unlabeled" in sections,
            "has_conclusion": "conclusion" in sections,
            "has_references": "references" in sections,
        }


def parse_paper_sections(
    text: str, parser: ResearchPaperSectionParser = None
) -> Tuple[Dict[str, Section], Dict]:
    """
    Parse sections from research paper text.

    Args:
        text: Cleaned text content
        parser: Optional parser instance

    Returns:
        Tuple of (sections_dict, summary_stats)
    """
    if parser is None:
        parser = ResearchPaperSectionParser()

    if not text or not text.strip():
        return {}, {"total_sections": 0, "total_words": 0, "sections_found": []}

    try:
        sections = parser.extract_sections(text)
        summary = parser.get_section_summary(sections)
        return sections, summary
    except Exception as e:
        logger.error(f"Section parsing failed: {e}")
        # Fallback: treat entire text as body
        fallback_section = Section(
            name="body",
            content=text.strip(),
            start_pos=0,
            end_pos=len(text),
            level=1,
            confidence=0.5,
        )
        return {"body": fallback_section}, {
            "total_sections": 1,
            "total_words": len(text.split()),
            "sections_found": ["body"],
        }
