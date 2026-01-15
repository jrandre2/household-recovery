# PDF Retrieval Module

Local PDF document processing.

```python
from household_recovery.pdf_retrieval import (
    PDFReader,
    LocalPaper,
    LocalPaperRetriever,
    load_recovery_papers,
    extract_abstract,
    DISASTER_RECOVERY_KEYWORDS
)
```

## Overview

This module enables extracting heuristics from local PDF files, allowing you to use your own research library instead of (or in addition to) Google Scholar.

---

## LocalPaper

Represents a locally stored academic paper.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Paper title (from filename) |
| `abstract` | `str` | Extracted abstract (~2000 chars) |
| `filepath` | `Path` | Path to PDF file |
| `full_text` | `str` | Full extracted text |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `authors` | `str` | Returns "Local paper" |
| `year` | `str` | Returns "Unknown" |

---

## PDFReader

Reads and extracts text from PDF files.

Supports both `pypdf` (recommended) and `PyPDF2` libraries.

### Constructor

```python
reader = PDFReader()
```

### Methods

#### `extract_text(filepath, max_pages=5) -> str`

Extract text from a PDF file.

```python
reader = PDFReader()
text = reader.extract_text(
    filepath=Path("paper.pdf"),
    max_pages=5  # Only read first 5 pages for efficiency
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `Path` | required | Path to PDF file |
| `max_pages` | `int` | 5 | Maximum pages to read |

---

## LocalPaperRetriever

Retrieves papers from a local PDF directory.

### Constructor

```python
retriever = LocalPaperRetriever(pdf_dir="/path/to/papers")
```

### Methods

#### `list_papers() -> list[Path]`

List all PDF files in the directory.

```python
papers = retriever.list_papers()
print(f"Found {len(papers)} PDFs")
```

#### `filter_relevant(keywords=None, max_papers=10) -> list[Path]`

Filter PDFs by filename keywords.

```python
papers = retriever.filter_relevant(
    keywords=['recovery', 'disaster', 'resilience'],
    max_papers=5
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keywords` | `list[str]` | `None` | Keywords to match (case-insensitive) |
| `max_papers` | `int` | 10 | Maximum papers to return |

#### `load_papers(pdf_paths=None, keywords=None, max_papers=5) -> list[LocalPaper]`

Load papers from PDF files.

```python
papers = retriever.load_papers(
    keywords=['recovery', 'housing'],
    max_papers=5
)

for paper in papers:
    print(f"Title: {paper.title}")
    print(f"Abstract: {paper.abstract[:200]}...")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_paths` | `list[Path]` | `None` | Specific paths to load |
| `keywords` | `list[str]` | `None` | Filter by filename |
| `max_papers` | `int` | 5 | Maximum to load |

---

## extract_abstract

Extract the abstract section from paper text.

```python
abstract = extract_abstract(
    full_text="...",
    max_length=2000
)
```

Tries to find explicit "Abstract" section markers. Falls back to first chunk of text if no abstract is found.

---

## load_recovery_papers

Convenience function to load disaster recovery papers.

```python
papers = load_recovery_papers(
    pdf_dir="/path/to/papers",
    max_papers=5
)
```

First tries to find papers matching disaster recovery keywords, then falls back to any available PDFs.

---

## DISASTER_RECOVERY_KEYWORDS

Default keywords for finding relevant papers.

```python
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
```

---

## Installation

Install PDF reading library:

```bash
pip install pypdf
# or
pip install PyPDF2
```

---

## Example: Using Local PDFs for Heuristics

```python
from household_recovery.heuristics import build_knowledge_base_from_pdfs
from household_recovery import SimulationEngine, SimulationConfig

# Build heuristics from local research library
heuristics = build_knowledge_base_from_pdfs(
    pdf_dir="/home/user/research/disaster-papers",
    groq_api_key="YOUR_GROQ_KEY",
    keywords=['recovery', 'household', 'community'],
    num_papers=5
)

print(f"Extracted {len(heuristics)} heuristics from PDFs")
for h in heuristics:
    print(f"  IF {h.condition_str} THEN {h.action}")

# Run simulation with extracted heuristics
config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config, heuristics=heuristics)
result = engine.run()
```

---

## Command Line Usage

```bash
# Use local PDFs for heuristic extraction
python -m household_recovery \
    --pdf-dir ~/research/papers \
    --groq-key YOUR_GROQ_KEY \
    --households 50 \
    --steps 20
```

---

## Hybrid Mode: PDFs + Scholar

```python
from household_recovery.heuristics import build_knowledge_base_hybrid

heuristics = build_knowledge_base_hybrid(
    pdf_dir="/path/to/local/papers",
    serpapi_key="YOUR_SERPAPI_KEY",
    groq_api_key="YOUR_GROQ_KEY",
    scholar_query="disaster recovery household resilience",
    num_papers=5,
    prefer_local=True  # Try local PDFs first
)
```

If local PDFs don't yield heuristics, automatically falls back to Google Scholar.
