"""
PDF-based document retrieval for local academic papers.

This module allows extracting heuristics from local PDF files,
enabling the simulation to use your own research library instead
of (or in addition to) Google Scholar.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# US-based filtering helpers
_US_STRONG_PATTERNS = [
    re.compile(r"\bUnited States of America\b", re.IGNORECASE),
    re.compile(r"\bUnited States\b", re.IGNORECASE),
    re.compile(r"\bU\.S\.A\.?\b", re.IGNORECASE),
    re.compile(r"\bU\.S\.\b", re.IGNORECASE),
    re.compile(r"\bUSA\b", re.IGNORECASE),
    re.compile(r"\bFEMA\b", re.IGNORECASE),
    re.compile(r"\bSBA\b", re.IGNORECASE),
    re.compile(r"\bCDBG\b", re.IGNORECASE),
    re.compile(r"\bNFIP\b", re.IGNORECASE),
    re.compile(r"\bHUD\b", re.IGNORECASE),
    re.compile(r"\bUSGS\b", re.IGNORECASE),
    re.compile(r"\bNOAA\b", re.IGNORECASE),
]

_US_STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia", "Washington DC", "Washington, DC", "Puerto Rico",
]

_US_STATE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in _US_STATE_NAMES) + r")\b",
    re.IGNORECASE,
)


def is_us_based(text: str) -> bool:
    """Return True if the text suggests the study is US-based."""
    if not text:
        return False
    for pattern in _US_STRONG_PATTERNS:
        if pattern.search(text):
            return True
    return _US_STATE_PATTERN.search(text) is not None


@dataclass
class LocalPaper:
    """Represents a locally stored academic paper."""
    title: str
    abstract: str  # First ~2000 chars or extracted abstract
    filepath: Path
    full_text: str = ""

    @property
    def authors(self) -> str:
        return "Local paper"

    @property
    def year(self) -> str:
        return "Unknown"


class PDFReader:
    """Reads and extracts text from PDF files."""

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if PDF reading libraries are available."""
        try:
            import pypdf
            self._has_pypdf = True
        except ImportError:
            self._has_pypdf = False

        try:
            import PyPDF2
            self._has_pypdf2 = True
        except ImportError:
            self._has_pypdf2 = False

        if self._has_pypdf or self._has_pypdf2:
            return True

        logger.error("No PDF library found. Install with: pip install pypdf")
        return False

    def extract_text(self, filepath: Path, max_pages: int | None = 5) -> str:
        """
        Extract text from a PDF file.

        Args:
            filepath: Path to the PDF file
            max_pages: Maximum number of pages to read (None = all pages)

        Returns:
            Extracted text content
        """
        if getattr(self, "_has_pypdf", False):
            try:
                return self._extract_with_pypdf(filepath, max_pages)
            except Exception as e:
                logger.warning(f"Failed to extract text with pypdf from {filepath}: {e}")
                if getattr(self, "_has_pypdf2", False):
                    try:
                        return self._extract_with_pypdf2(filepath, max_pages)
                    except Exception as e2:
                        logger.warning(f"Failed to extract text with PyPDF2 from {filepath}: {e2}")
                return ""

        if getattr(self, "_has_pypdf2", False):
            try:
                return self._extract_with_pypdf2(filepath, max_pages)
            except Exception as e:
                logger.warning(f"Failed to extract text with PyPDF2 from {filepath}: {e}")
                return ""

        logger.warning(f"No PDF backend available to read {filepath}")
        return ""

    def _extract_with_pypdf(self, filepath: Path, max_pages: int | None) -> str:
        """Extract text using pypdf library."""
        import pypdf

        text_parts = []
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f, strict=False)
            num_pages = len(reader.pages) if max_pages is None else min(len(reader.pages), max_pages)

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, filepath: Path, max_pages: int | None) -> str:
        """Extract text using PyPDF2 library."""
        import PyPDF2

        text_parts = []
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f, strict=False)
            num_pages = len(reader.pages) if max_pages is None else min(len(reader.pages), max_pages)

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)


def extract_abstract(full_text: str, max_length: int = 2000) -> str:
    """
    Extract the abstract section from paper text.

    Tries to find explicit "Abstract" section, otherwise returns
    the first chunk of text.
    """
    text_lower = full_text.lower()

    # Try to find abstract section
    abstract_markers = ['abstract', 'summary']
    end_markers = ['introduction', 'keywords', 'key words', '1.', '1 ']

    for marker in abstract_markers:
        start_idx = text_lower.find(marker)
        if start_idx != -1:
            # Found abstract marker, find where it ends
            search_start = start_idx + len(marker)
            end_idx = len(full_text)

            for end_marker in end_markers:
                end_pos = text_lower.find(end_marker, search_start)
                if end_pos != -1 and end_pos < end_idx:
                    end_idx = end_pos

            abstract = full_text[start_idx:end_idx].strip()

            # Clean up
            if abstract.lower().startswith('abstract'):
                abstract = abstract[8:].strip()
            if abstract.startswith(':') or abstract.startswith('.'):
                abstract = abstract[1:].strip()

            if len(abstract) > 100:  # Reasonable abstract length
                return abstract[:max_length]

    # Fallback: return first chunk of text
    return full_text[:max_length]


class LocalPaperRetriever:
    """Retrieves papers from a local PDF directory."""

    def __init__(self, pdf_dir: Path | str):
        self.pdf_dir = Path(pdf_dir)
        self.reader = PDFReader()

    def list_papers(self) -> list[Path]:
        """List all PDF files in the directory."""
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
            return []

        return sorted(self.pdf_dir.glob("*.pdf"))

    def filter_relevant(
        self,
        keywords: list[str] | None = None,
        max_papers: int | None = 10,
    ) -> list[Path]:
        """
        Filter PDFs by filename keywords.

        Args:
            keywords: List of keywords to match in filename (case-insensitive)
            max_papers: Maximum number of papers to return (None for no limit)

        Returns:
            List of matching PDF paths
        """
        all_papers = self.list_papers()

        if not keywords:
            return all_papers if max_papers is None else all_papers[:max_papers]

        keywords_lower = [k.lower() for k in keywords]

        matching = []
        for pdf_path in all_papers:
            name_lower = pdf_path.stem.lower()
            if any(kw in name_lower for kw in keywords_lower):
                matching.append(pdf_path)
                if max_papers is not None and len(matching) >= max_papers:
                    break

        return matching

    def load_papers(
        self,
        pdf_paths: list[Path] | None = None,
        keywords: list[str] | None = None,
        max_papers: int = 5,
        max_pages: int | None = 5,
        us_only: bool = False,
    ) -> list[LocalPaper]:
        """
        Load papers from PDF files.

        Args:
            pdf_paths: Specific PDF paths to load (overrides keywords)
            keywords: Keywords to filter by filename
            max_papers: Maximum papers to load
            max_pages: Maximum pages per PDF to read (None = all pages)
            us_only: If True, only include US-based studies

        Returns:
            List of LocalPaper objects with extracted text
        """
        if pdf_paths is None:
            if keywords:
                pdf_paths = self.filter_relevant(
                    keywords,
                    None if us_only else max_papers,
                )
            else:
                pdf_paths = self.list_papers()
                if not us_only:
                    pdf_paths = pdf_paths[:max_papers]

        papers = []
        for pdf_path in pdf_paths:
            logger.info(f"Reading: {pdf_path.name}")

            full_text = self.reader.extract_text(pdf_path, max_pages=max_pages)
            if not full_text:
                logger.warning(f"  No text extracted, skipping")
                continue

            abstract = extract_abstract(full_text)

            # Use filename as title (clean it up)
            title = pdf_path.stem.replace("-", " ").replace("_", " ")

            focus_text = full_text[:8000]
            text_for_filter = f"{pdf_path.name}\n{title}\n{abstract}\n{focus_text}"
            if us_only and not is_us_based(text_for_filter):
                logger.info("  Skipping non-US paper")
                continue

            paper = LocalPaper(
                title=title,
                abstract=abstract,
                filepath=pdf_path,
                full_text=full_text
            )
            papers.append(paper)

            logger.info(f"  Extracted {len(abstract)} chars of abstract")
            if max_papers is not None and len(papers) >= max_papers:
                break

        return papers


# Keywords for finding disaster recovery papers
DISASTER_RECOVERY_KEYWORDS = [
    'recovery',
    'disaster',
    'housing',
    'resilience',
    'flood',
    'earthquake',
    'hurricane',
    'reconstruction',
    'post-disaster',
    'vulnerability',
    'community',
]


def load_recovery_papers(
    pdf_dir: Path | str,
    max_papers: int = 5,
    max_pages: int | None = 5,
    us_only: bool = False,
) -> list[LocalPaper]:
    """
    Convenience function to load disaster recovery papers from a directory.

    Args:
        pdf_dir: Directory containing PDF files
        max_papers: Maximum number of papers to load
        max_pages: Maximum pages per PDF to read (None = all pages)
        us_only: If True, only include US-based studies

    Returns:
        List of LocalPaper objects
    """
    retriever = LocalPaperRetriever(pdf_dir)

    # First try to find relevant papers by keywords
    papers = retriever.load_papers(
        keywords=DISASTER_RECOVERY_KEYWORDS,
        max_papers=max_papers,
        max_pages=max_pages,
        us_only=us_only,
    )

    if not papers:
        # Fallback to any PDFs
        logger.info("No keyword matches, loading first available PDFs")
        papers = retriever.load_papers(
            max_papers=max_papers,
            max_pages=max_pages,
            us_only=us_only,
        )

    return papers
