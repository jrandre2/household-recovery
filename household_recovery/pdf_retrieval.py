"""
PDF-based document retrieval for local academic papers.

This module allows extracting heuristics from local PDF files,
enabling the simulation to use your own research library instead
of (or in addition to) Google Scholar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


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
            self._use_pypdf = True
            return True
        except ImportError:
            pass

        try:
            import PyPDF2
            self._use_pypdf = False
            return True
        except ImportError:
            pass

        logger.error(
            "No PDF library found. Install with: pip install pypdf"
        )
        return False

    def extract_text(self, filepath: Path, max_pages: int = 5) -> str:
        """
        Extract text from a PDF file.

        Args:
            filepath: Path to the PDF file
            max_pages: Maximum number of pages to read (for efficiency)

        Returns:
            Extracted text content
        """
        try:
            if hasattr(self, '_use_pypdf') and self._use_pypdf:
                return self._extract_with_pypdf(filepath, max_pages)
            else:
                return self._extract_with_pypdf2(filepath, max_pages)
        except Exception as e:
            logger.warning(f"Failed to extract text from {filepath}: {e}")
            return ""

    def _extract_with_pypdf(self, filepath: Path, max_pages: int) -> str:
        """Extract text using pypdf library."""
        import pypdf

        text_parts = []
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            num_pages = min(len(reader.pages), max_pages)

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, filepath: Path, max_pages: int) -> str:
        """Extract text using PyPDF2 library."""
        import PyPDF2

        text_parts = []
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = min(len(reader.pages), max_pages)

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
        max_papers: int = 10
    ) -> list[Path]:
        """
        Filter PDFs by filename keywords.

        Args:
            keywords: List of keywords to match in filename (case-insensitive)
            max_papers: Maximum number of papers to return

        Returns:
            List of matching PDF paths
        """
        all_papers = self.list_papers()

        if not keywords:
            return all_papers[:max_papers]

        keywords_lower = [k.lower() for k in keywords]

        matching = []
        for pdf_path in all_papers:
            name_lower = pdf_path.stem.lower()
            if any(kw in name_lower for kw in keywords_lower):
                matching.append(pdf_path)
                if len(matching) >= max_papers:
                    break

        return matching

    def load_papers(
        self,
        pdf_paths: list[Path] | None = None,
        keywords: list[str] | None = None,
        max_papers: int = 5
    ) -> list[LocalPaper]:
        """
        Load papers from PDF files.

        Args:
            pdf_paths: Specific PDF paths to load (overrides keywords)
            keywords: Keywords to filter by filename
            max_papers: Maximum papers to load

        Returns:
            List of LocalPaper objects with extracted text
        """
        if pdf_paths is None:
            if keywords:
                pdf_paths = self.filter_relevant(keywords, max_papers)
            else:
                pdf_paths = self.list_papers()[:max_papers]

        papers = []
        for pdf_path in pdf_paths:
            logger.info(f"Reading: {pdf_path.name}")

            full_text = self.reader.extract_text(pdf_path)
            if not full_text:
                logger.warning(f"  No text extracted, skipping")
                continue

            abstract = extract_abstract(full_text)

            # Use filename as title (clean it up)
            title = pdf_path.stem.replace("-", " ").replace("_", " ")

            paper = LocalPaper(
                title=title,
                abstract=abstract,
                filepath=pdf_path,
                full_text=full_text
            )
            papers.append(paper)

            logger.info(f"  Extracted {len(abstract)} chars of abstract")

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
    max_papers: int = 5
) -> list[LocalPaper]:
    """
    Convenience function to load disaster recovery papers from a directory.

    Args:
        pdf_dir: Directory containing PDF files
        max_papers: Maximum number of papers to load

    Returns:
        List of LocalPaper objects
    """
    retriever = LocalPaperRetriever(pdf_dir)

    # First try to find relevant papers by keywords
    papers = retriever.load_papers(
        keywords=DISASTER_RECOVERY_KEYWORDS,
        max_papers=max_papers
    )

    if not papers:
        # Fallback to any PDFs
        logger.info("No keyword matches, loading first available PDFs")
        papers = retriever.load_papers(max_papers=max_papers)

    return papers
