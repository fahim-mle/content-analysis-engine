# phase_clean/citation_handler.py
"""Citation extraction, parsing, and normalization for research papers."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """Represents a single citation."""

    raw_text: str
    authors: List[str]
    title: Optional[str]
    year: Optional[int]
    venue: Optional[str]  # Journal, conference, etc.
    volume: Optional[str]
    pages: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
    citation_type: str  # 'journal', 'conference', 'arxiv', 'book', 'misc'
    confidence: float  # How confident we are in the parsing


class CitationHandler:
    """Extract and normalize citations from research papers."""

    def __init__(self):
        # Common citation patterns
        self.patterns = {
            # Author patterns
            "author_name": re.compile(r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)"),
            "author_last_first": re.compile(r"([A-Z][a-z]+,\s*[A-Z]\.?\s*[A-Z]?\.?)"),
            "author_initials": re.compile(r"([A-Z]\.?\s*[A-Z]\.?\s*[A-Z][a-z]+)"),
            # Year patterns
            "year_in_parens": re.compile(r"\((\d{4})\)"),
            "year_standalone": re.compile(r"\b(19|20)\d{2}\b"),
            # Title patterns (quoted or capitalized)
            "quoted_title": re.compile(r'"([^"]{10,200})"'),
            "capitalized_title": re.compile(r"\b([A-Z][^.!?]*[.!?])"),
            # Venue patterns
            "journal_pattern": re.compile(
                r"\b(Journal|Proceedings|IEEE|ACM|Nature|Science)\b", re.IGNORECASE
            ),
            "conference_pattern": re.compile(
                r"\b(Conference|Workshop|Symposium|ICML|NeurIPS|ICLR|AAAI|IJCAI)\b",
                re.IGNORECASE,
            ),
            # Volume and pages
            "volume_pages": re.compile(r"(\d+)\s*\((\d+)\):\s*(\d+[-–]\d+)"),
            "volume_simple": re.compile(r"vol\.?\s*(\d+)", re.IGNORECASE),
            "pages_range": re.compile(r"pp?\.?\s*(\d+[-–]\d+)", re.IGNORECASE),
            # DOI and URLs
            "doi": re.compile(r"doi:\s*(10\.\d+/[^\s]+)", re.IGNORECASE),
            "arxiv": re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE),
            "url": re.compile(r"https?://[^\s]+"),
            # Common reference list indicators
            "reference_markers": re.compile(r"^\s*[\[\(]?\d+[\]\)]?\s*", re.MULTILINE),
        }

        # Common academic venues for classification
        self.journal_keywords = {
            "journal",
            "transactions",
            "proceedings",
            "review",
            "letters",
            "ieee",
            "acm",
            "nature",
            "science",
            "cell",
            "lancet",
        }

        self.conference_keywords = {
            "conference",
            "workshop",
            "symposium",
            "meeting",
            "icml",
            "neurips",
            "nips",
            "iclr",
            "aaai",
            "ijcai",
            "acl",
            "emnlp",
            "cvpr",
            "iccv",
            "eccv",
            "sigir",
            "www",
            "chi",
        }

    def extract_reference_section(self, text: str) -> Optional[str]:
        """Extract the references section from the paper text."""
        if not text:
            return None

        # Look for references section header
        ref_patterns = [
            r"\n\s*references?\s*\n",
            r"\n\s*bibliography\s*\n",
            r"\n\s*\d+\.?\s*references?\s*\n",
            r"\n\s*\d+\.?\s*bibliography\s*\n",
        ]

        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ref_start = match.end()
                # Extract everything after references section
                ref_text = text[ref_start:].strip()

                # Remove potential appendix or other sections after references
                appendix_match = re.search(
                    r"\n\s*appendix\s*\n", ref_text, re.IGNORECASE
                )
                if appendix_match:
                    ref_text = ref_text[: appendix_match.start()]

                return ref_text

        # If no clear references section, look for citation-like patterns at the end
        lines = text.split("\n")
        potential_refs = []
        in_refs = False

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line looks like a reference
            if (
                re.match(r"^\s*[\[\(]?\d+[\]\)]?", line)
                or re.search(r"\(\d{4}\)", line)
                or "doi:" in line.lower()
                or "arxiv:" in line.lower()
            ):
                potential_refs.append(line)
                in_refs = True
            elif in_refs and len(potential_refs) > 3:
                # Stop if we've found several refs and hit non-ref content
                break
            elif in_refs:
                potential_refs = []  # Reset if we haven't found enough refs
                in_refs = False

        if len(potential_refs) >= 3:
            return "\n".join(reversed(potential_refs))

        return None

    def parse_single_citation(self, citation_text: str) -> Citation:
        """Parse a single citation text into structured data."""
        citation_text = citation_text.strip()

        # Initialize citation object
        citation = Citation(
            raw_text=citation_text,
            authors=[],
            title=None,
            year=None,
            venue=None,
            volume=None,
            pages=None,
            doi=None,
            arxiv_id=None,
            url=None,
            citation_type="misc",
            confidence=0.5,
        )

        try:
            # Extract year
            year_match = self.patterns["year_in_parens"].search(citation_text)
            if year_match:
                citation.year = int(year_match.group(1))
                citation.confidence += 0.2
            else:
                year_matches = self.patterns["year_standalone"].findall(citation_text)
                if year_matches:
                    citation.year = int(year_matches[-1])  # Take the last year found
                    citation.confidence += 0.1

            # Extract DOI
            doi_match = self.patterns["doi"].search(citation_text)
            if doi_match:
                citation.doi = doi_match.group(1)
                citation.confidence += 0.3

            # Extract ArXiv ID
            arxiv_match = self.patterns["arxiv"].search(citation_text)
            if arxiv_match:
                citation.arxiv_id = arxiv_match.group(1)
                citation.citation_type = "arxiv"
                citation.confidence += 0.3

            # Extract URL
            url_match = self.patterns["url"].search(citation_text)
            if url_match:
                citation.url = url_match.group(0)
                citation.confidence += 0.1

            # Extract title (quoted or well-formed)
            title_match = self.patterns["quoted_title"].search(citation_text)
            if title_match:
                citation.title = title_match.group(1).strip()
                citation.confidence += 0.2

            # Extract authors (simplified approach)
            # Remove known non-author parts first
            author_text = citation_text
            if citation.year:
                author_text = re.sub(rf"\({citation.year}\)", "", author_text)
            if citation.title:
                author_text = author_text.replace(f'"{citation.title}"', "")

            # Look for author patterns
            author_matches = self.patterns["author_name"].findall(
                author_text[:200]
            )  # First 200 chars
            if author_matches:
                citation.authors = [
                    name.strip() for name in author_matches[:5]
                ]  # Max 5 authors
                citation.confidence += 0.2

            # Extract venue info and classify citation type
            venue_text = citation_text.lower()

            # Check for journal indicators
            if any(keyword in venue_text for keyword in self.journal_keywords):
                citation.citation_type = "journal"
                citation.confidence += 0.1

            # Check for conference indicators
            elif any(keyword in venue_text for keyword in self.conference_keywords):
                citation.citation_type = "conference"
                citation.confidence += 0.1

            # Extract volume and pages
            vol_pages_match = self.patterns["volume_pages"].search(citation_text)
            if vol_pages_match:
                citation.volume = vol_pages_match.group(1)
                citation.pages = vol_pages_match.group(3)
                citation.confidence += 0.1
            else:
                vol_match = self.patterns["volume_simple"].search(citation_text)
                if vol_match:
                    citation.volume = vol_match.group(1)

                pages_match = self.patterns["pages_range"].search(citation_text)
                if pages_match:
                    citation.pages = pages_match.group(1)

        except Exception as e:
            logger.debug(f"Error parsing citation: {e}")
            citation.confidence = 0.1

        return citation

    def extract_citations(self, references_text: str) -> List[Citation]:
        """Extract and parse all citations from references text."""
        if not references_text:
            return []

        citations = []

        # Split references by common patterns
        # Look for numbered references [1], (1), or standalone numbers
        ref_splits = re.split(r"\n\s*[\[\(]?\d+[\]\)]?\s*", references_text)

        if len(ref_splits) < 2:
            # Try alternative splitting - by double newlines
            ref_splits = references_text.split("\n\n")

        for ref_text in ref_splits:
            ref_text = ref_text.strip()
            if len(ref_text) < 20:  # Skip very short entries
                continue

            # Remove reference numbers at the start
            ref_text = re.sub(r"^\s*[\[\(]?\d+[\]\)]?\s*", "", ref_text)

            if ref_text:
                citation = self.parse_single_citation(ref_text)
                citations.append(citation)

        return citations

    def extract_inline_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extract inline citations from main text (e.g., [1], (Smith, 2020))."""
        inline_citations = []

        # Pattern for numbered citations [1], [2,3], etc.
        numbered_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
        for match in re.finditer(numbered_pattern, text):
            citation_nums = [int(x.strip()) for x in match.group(1).split(",")]
            for num in citation_nums:
                inline_citations.append((f"[{num}]", match.start()))

        # Pattern for author-year citations (Smith, 2020), (Smith et al., 2020)
        author_year_pattern = r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)"
        for match in re.finditer(author_year_pattern, text):
            citation_text = match.group(0)
            inline_citations.append((citation_text, match.start()))

        return inline_citations

    def normalize_citations(self, citations: List[Citation]) -> List[Dict]:
        """Normalize citations to a standard format."""
        normalized = []

        for i, citation in enumerate(citations):
            normalized_citation = {
                "id": i + 1,
                "raw_text": citation.raw_text,
                "authors": citation.authors,
                "title": citation.title,
                "year": citation.year,
                "venue": citation.venue,
                "volume": citation.volume,
                "pages": citation.pages,
                "doi": citation.doi,
                "arxiv_id": citation.arxiv_id,
                "url": citation.url,
                "type": citation.citation_type,
                "confidence": round(citation.confidence, 2),
                "is_valid": citation.confidence > 0.3
                and (citation.year is not None or citation.arxiv_id is not None),
            }
            normalized.append(normalized_citation)

        return normalized

    def clean_references_section(self, references_text: str) -> str:
        """Clean and normalize the references section text."""
        if not references_text:
            return ""

        # Remove excessive whitespace
        references_text = re.sub(r"\s+", " ", references_text)

        # Normalize line breaks around reference entries
        references_text = re.sub(r"\s*\n\s*", "\n", references_text)

        # Remove very short lines (likely page numbers or artifacts)
        lines = references_text.split("\n")
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]

        return "\n".join(cleaned_lines)

    def get_citation_stats(self, citations: List[Citation]) -> Dict:
        """Generate statistics about the extracted citations."""
        if not citations:
            return {
                "total_citations": 0,
                "valid_citations": 0,
                "avg_confidence": 0.0,
                "types": {},
                "years_range": None,
            }

        valid_citations = [c for c in citations if c.confidence > 0.3]

        # Count citation types
        type_counts = {}
        for citation in citations:
            type_counts[citation.citation_type] = (
                type_counts.get(citation.citation_type, 0) + 1
            )

        # Get year range
        years = [c.year for c in citations if c.year is not None]
        years_range = (min(years), max(years)) if years else None

        return {
            "total_citations": len(citations),
            "valid_citations": len(valid_citations),
            "avg_confidence": round(
                sum(c.confidence for c in citations) / len(citations), 2
            ),
            "types": type_counts,
            "years_range": years_range,
            "has_arxiv": any(c.arxiv_id for c in citations),
            "has_doi": any(c.doi for c in citations),
        }


def process_citations(
    text: str, handler: CitationHandler = None
) -> Tuple[List[Dict], Dict, str]:
    """
    Process citations from paper text.

    Args:
        text: Full paper text
        handler: Optional citation handler instance

    Returns:
        Tuple of (normalized_citations, citation_stats, cleaned_references_text)
    """
    if handler is None:
        handler = CitationHandler()

    if not text or not text.strip():
        return [], {"total_citations": 0}, ""

    try:
        # Extract references section
        references_text = handler.extract_reference_section(text)

        if not references_text:
            logger.debug("No references section found")
            return [], {"total_citations": 0}, ""

        # Clean references text
        cleaned_references = handler.clean_references_section(references_text)

        # Extract and parse citations
        citations = handler.extract_citations(cleaned_references)

        # Normalize citations
        normalized_citations = handler.normalize_citations(citations)

        # Generate statistics
        stats = handler.get_citation_stats(citations)

        logger.debug(
            f"Extracted {len(citations)} citations, {stats['valid_citations']} valid"
        )

        return normalized_citations, stats, cleaned_references

    except Exception as e:
        logger.error(f"Citation processing failed: {e}")
        return [], {"total_citations": 0, "error": str(e)}, ""
