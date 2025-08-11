# phase_clean/quality_filter.py
"""Text quality assessment and filtering for research papers."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality assessment metrics for research paper text."""

    length_score: float
    readability_score: float
    structure_score: float
    language_score: float
    completeness_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]


class TextQualityFilter:
    """Assess and filter research paper text based on quality metrics."""

    def __init__(self):
        from .config import MIN_QUALITY_SCORE, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, TARGET_CATEGORIES
        self.min_quality_score = MIN_QUALITY_SCORE
        self.min_text_length = MIN_TEXT_LENGTH
        self.max_text_length = MAX_TEXT_LENGTH
        self.target_categories = TARGET_CATEGORIES

        # Language detection patterns
        self.english_patterns = {
            "common_words": re.compile(
                r"\b(the|and|of|to|a|in|is|it|you|that|he|was|for|on|are|as|with|his|they|i|at|be|this|have|from|or|one|had|by|word|but|what|some|we|can|out|other|were|all|there|when|up|use|your|how|said|each|which|their|time|will|about|if|up|out|many|then|them|these|so|some|her|would|make|like|into|him|has|two|more|very|what|know|just|first|get|over|think|also|do)\b",
                re.IGNORECASE,
            ),
            "articles": re.compile(r"\b(a|an|the)\b", re.IGNORECASE),
            "prepositions": re.compile(
                r"\b(in|on|at|by|for|with|without|through|during|before|after|above|below|up|down|out|off|over|under|again|further|then|once)\b",
                re.IGNORECASE,
            ),
        }

        # Academic writing indicators
        self.academic_indicators = {
            "methodology_terms": re.compile(
                r"\b(method|approach|algorithm|technique|framework|model|system|analysis|evaluation|experiment|study|research|data|results|findings|conclusion|abstract|introduction)\b",
                re.IGNORECASE,
            ),
            "research_verbs": re.compile(
                r"\b(propose|present|demonstrate|show|prove|analyze|evaluate|compare|investigate|explore|examine|study|develop|implement|design)\b",
                re.IGNORECASE,
            ),
            "academic_phrases": re.compile(
                r"\b(et al\.|i\.e\.|e\.g\.|cf\.|viz\.|ibid\.|op\. cit\.)\b",
                re.IGNORECASE,
            ),
        }

        # Quality issue patterns
        self.issue_patterns = {
            "excessive_numbers": re.compile(r"\b\d+\b"),
            "special_chars": re.compile(r"[^\w\s\.\,\!\?\:\;\"\'\(\)\[\]\{\}\-]"),
            "repeated_chars": re.compile(r"(.)\1{4,}"),  # 5+ repeated chars
            "broken_words": re.compile(
                r"\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b){3,}"
            ),  # Many short words in sequence
            "page_artifacts": re.compile(r"^\s*\d+\s*$", re.MULTILINE),
            "url_artifacts": re.compile(r"https?://\S+|www\.\S+"),
        }

    def assess_length_quality(self, text: str) -> Tuple[float, List[str]]:
        """Assess text length quality."""
        issues = []
        score = 1.0

        length = len(text)
        word_count = len(text.split())

        if length < self.min_text_length:
            issues.append(f"text_too_short_{length}_chars")
            score = 0.0
        elif length < 1000:
            issues.append("text_short")
            score = 0.4
        elif length > self.max_text_length:
            issues.append(f"text_too_long_{length}_chars")
            score = 0.6
        elif length > 50000:
            score = 0.8  # Very long but acceptable

        # Check word count
        if word_count < 100:
            issues.append(f"low_word_count_{word_count}")
            score = min(score, 0.3)

        return score, issues

    def assess_readability_quality(self, text: str) -> Tuple[float, List[str]]:
        """Assess text readability and language quality."""
        issues = []
        score = 1.0

        if not text.strip():
            return 0.0, ["empty_text"]

        words = text.split()
        chars = len(text)

        # Check character-to-word ratio
        avg_word_length = chars / max(len(words), 1)
        if avg_word_length < 3:
            issues.append("very_short_words")
            score *= 0.6
        elif avg_word_length > 15:
            issues.append("very_long_words")  # Possible encoding issues
            score *= 0.7

        # Check for English language indicators
        english_score = 0.0
        total_checks = len(self.english_patterns)

        for pattern_name, pattern in self.english_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                english_score += 1.0 / total_checks

        if english_score < 0.5:
            issues.append("likely_non_english")
            score *= 0.2
        elif english_score < 0.7:
            issues.append("possibly_non_english")
            score *= 0.6

        # Check for excessive special characters
        special_char_count = len(self.issue_patterns["special_chars"].findall(text))
        special_char_ratio = special_char_count / max(chars, 1)
        if special_char_ratio > 0.1:
            issues.append(f"excessive_special_chars_{special_char_ratio:.2f}")
            score *= 0.5

        # Check for repeated character patterns (OCR errors)
        repeated_matches = self.issue_patterns["repeated_chars"].findall(text)
        if len(repeated_matches) > 5:
            issues.append("repeated_char_patterns")
            score *= 0.7

        return score, issues

    def assess_structure_quality(
        self, text: str, sections: Dict = None
    ) -> Tuple[float, List[str]]:
        """Assess document structure quality."""
        issues = []
        score = 0.5  # Base score

        # Check for academic structure indicators
        academic_score = 0.0
        total_indicators = len(self.academic_indicators)

        for indicator_name, pattern in self.academic_indicators.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                academic_score += 1.0 / total_indicators

        if academic_score > 0.8:
            score += 0.4
        elif academic_score > 0.5:
            score += 0.3
        elif academic_score > 0.2:
            score += 0.1
        else:
            issues.append("lacks_academic_indicators")

        # Check section structure if available
        if sections:
            section_score = 0.0
            important_sections = [
                "abstract",
                "introduction",
                "methodology",
                "results",
                "conclusion",
            ]

            for section in important_sections:
                if section in sections:
                    section_score += 0.2

            if section_score > 0.6:
                score = min(score + 0.3, 1.0)
            elif section_score > 0.2:
                score = min(score + 0.1, 1.0)
            else:
                issues.append("missing_key_sections")

        # Check for broken text patterns
        broken_word_matches = self.issue_patterns["broken_words"].findall(text)
        if len(broken_word_matches) > 10:
            issues.append("broken_word_patterns")
            score *= 0.7

        # Check for excessive page artifacts
        page_artifacts = self.issue_patterns["page_artifacts"].findall(text)
        if len(page_artifacts) > 20:
            issues.append("excessive_page_numbers")
            score *= 0.8

        return score, issues

    def assess_completeness_quality(
        self, text: str, metadata: Dict = None
    ) -> Tuple[float, List[str]]:
        """Assess text completeness."""
        issues = []
        score = 1.0

        # Check for abrupt ending
        text_end = text[-200:].strip() if len(text) > 200 else text
        if not any(text_end.endswith(punct) for punct in [".", "!", "?", '"', "'"]):
            issues.append("abrupt_ending")
            score *= 0.8

        # Check for common truncation indicators
        truncation_indicators = [
            "...",
            "[truncated]",
            "[continued]",
            "see full text",
            "download full paper",
        ]

        text_lower = text.lower()
        for indicator in truncation_indicators:
            if indicator in text_lower:
                issues.append(f"truncation_indicator_{indicator}")
                score *= 0.3

        # Check if text seems too short for a research paper
        word_count = len(text.split())
        if word_count < 200:  # Very short, likely just title/abstract
            issues.append("likely_incomplete")
            score *= 0.4
        elif word_count < 500:  # Short but might be acceptable
            issues.append("possibly_incomplete")
            score *= 0.7

        return score, issues

    def assess_language_match(
        self, text: str, target_language: str = "en"
    ) -> Tuple[float, List[str]]:
        """Assess if text matches target language."""
        issues = []
        score = 1.0

        if target_language == "en":
            # English language assessment
            english_word_count = len(
                self.english_patterns["common_words"].findall(text)
            )
            total_words = len(text.split())

            if total_words > 0:
                english_ratio = english_word_count / total_words
                if english_ratio < 0.1:
                    issues.append(f"low_english_ratio_{english_ratio:.2f}")
                    score = 0.1
                elif english_ratio < 0.3:
                    issues.append(f"moderate_english_ratio_{english_ratio:.2f}")
                    score = 0.5

        return score, issues

    def filter_by_categories(self, categories: List[str]) -> Tuple[bool, List[str]]:
        """Check if paper categories match target categories."""
        issues = []

        if not categories:
            issues.append("no_categories")
            return False, issues

        # Check if any category matches our targets
        category_set = set(categories)
        if category_set.intersection(self.target_categories):
            return True, []

        # Check for related categories
        related_categories = {
            "cs.HC",
            "cs.IT",
            "cs.SD",
            "cs.SI",
            "stat.ML",
            "cs.DB",
            "cs.RO",
            "cs.CC",
            "cs.DC",
            "cs.DS",
            "cs.PL",
        }

        if category_set.intersection(related_categories):
            issues.append("related_categories_only")
            return True, issues  # Accept but note the issue

        issues.append(f"unrelated_categories_{','.join(categories)}")
        return False, issues

    def comprehensive_quality_assessment(
        self, text: str, metadata: Dict = None, sections: Dict = None
    ) -> QualityMetrics:
        """Perform comprehensive quality assessment."""

        # Individual assessments
        length_score, length_issues = self.assess_length_quality(text)
        readability_score, readability_issues = self.assess_readability_quality(text)
        structure_score, structure_issues = self.assess_structure_quality(
            text, sections
        )
        completeness_score, completeness_issues = self.assess_completeness_quality(
            text, metadata
        )
        language_score, language_issues = self.assess_language_match(text)

        # Combine all issues
        all_issues = (
            length_issues
            + readability_issues
            + structure_issues
            + completeness_issues
            + language_issues
        )

        # Calculate weighted overall score
        weights = {
            "length": 0.15,
            "readability": 0.25,
            "structure": 0.25,
            "completeness": 0.20,
            "language": 0.15,
        }

        overall_score = (
            length_score * weights["length"]
            + readability_score * weights["readability"]
            + structure_score * weights["structure"]
            + completeness_score * weights["completeness"]
            + language_score * weights["language"]
        )

        # Generate recommendations
        recommendations = []
        if length_score < 0.5:
            recommendations.append("Consider if text extraction was complete")
        if readability_score < 0.5:
            recommendations.append("Check for encoding or language issues")
        if structure_score < 0.5:
            recommendations.append("Verify academic paper structure")
        if completeness_score < 0.5:
            recommendations.append("Check for text truncation")

        return QualityMetrics(
            length_score=length_score,
            readability_score=readability_score,
            structure_score=structure_score,
            language_score=language_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            issues=all_issues,
            recommendations=recommendations,
        )

    def should_include_paper(
        self, quality_metrics: QualityMetrics, categories: List[str] = None
    ) -> Tuple[bool, str]:
        """Decide whether to include paper based on quality assessment."""

        # Check categories first
        if categories:
            category_ok, category_issues = self.filter_by_categories(categories)
            if not category_ok:
                return False, f"Category filter: {', '.join(category_issues)}"

        # Check overall quality
        if quality_metrics.overall_score < self.min_quality_score:
            return (
                False,
                f"Quality score {quality_metrics.overall_score:.2f} below threshold {self.min_quality_score}",
            )

        # Check for critical issues
        critical_issues = [
            "empty_text",
            "likely_non_english",
            "excessive_special_chars",
            "likely_incomplete",
        ]

        for issue in quality_metrics.issues:
            if any(critical in issue for critical in critical_issues):
                return False, f"Critical issue: {issue}"

        return True, "Passed quality filters"


def filter_papers_by_quality(
    papers: List[Dict], quality_filter: TextQualityFilter = None
) -> Tuple[List[Dict], Dict]:
    """
    Filter papers based on quality assessment.

    Args:
        papers: List of paper dictionaries with text data
        quality_filter: Optional filter instance

    Returns:
        Tuple of (filtered_papers, filter_stats)
    """
    if quality_filter is None:
        quality_filter = TextQualityFilter()

    if not papers:
        return [], {"total": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}}

    accepted_papers = []
    rejection_reasons = {}

    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "unknown")
        text = paper.get("cleaned_text", paper.get("full_text", ""))
        categories = paper.get("categories", [])
        sections = paper.get("text_sections", {})

        try:
            # Assess quality
            quality_metrics = quality_filter.comprehensive_quality_assessment(
                text, metadata=paper, sections=sections
            )

            # Add quality metrics to paper
            paper["quality_metrics"] = {
                "length_score": quality_metrics.length_score,
                "readability_score": quality_metrics.readability_score,
                "structure_score": quality_metrics.structure_score,
                "completeness_score": quality_metrics.completeness_score,
                "language_score": quality_metrics.language_score,
                "overall_score": quality_metrics.overall_score,
                "issues": quality_metrics.issues,
                "recommendations": quality_metrics.recommendations,
            }

            # Decide inclusion
            should_include, reason = quality_filter.should_include_paper(
                quality_metrics, categories
            )

            if should_include:
                accepted_papers.append(paper)
                logger.debug(
                    f"Accepted {arxiv_id}: quality score {quality_metrics.overall_score:.2f}"
                )
            else:
                reason_key = reason.split(":")[0] if ":" in reason else reason
                rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1
                logger.debug(f"Rejected {arxiv_id}: {reason}")

        except Exception as e:
            logger.error(f"Quality assessment failed for {arxiv_id}: {e}")
            rejection_reasons["processing_error"] = (
                rejection_reasons.get("processing_error", 0) + 1
            )

    stats = {
        "total": len(papers),
        "accepted": len(accepted_papers),
        "rejected": len(papers) - len(accepted_papers),
        "acceptance_rate": len(accepted_papers) / len(papers) if papers else 0.0,
        "rejection_reasons": rejection_reasons,
    }

    logger.info(
        f"Quality filter: {stats['accepted']}/{stats['total']} papers accepted ({stats['acceptance_rate']:.2%})"
    )

    return accepted_papers, stats
