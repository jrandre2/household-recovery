# Local PDFs Guide

Learn how to use your own research library for heuristic extraction.

## Overview

Instead of fetching papers from Google Scholar, you can extract heuristics from local PDF files. This is useful when:

- You have specific papers you want to use
- You're working offline
- You want full control over the research sources
- You have access to papers not available online

## Installation

Install PDF reading library:

```bash
pip install pypdf
```

## Basic Usage

### Command Line

```bash
python -m household_recovery \
    --pdf-dir ~/research/disaster-papers \
    --groq-key YOUR_GROQ_KEY \
    --households 50
```

### Python API

```python
from household_recovery.heuristics import build_knowledge_base_from_pdfs

heuristics = build_knowledge_base_from_pdfs(
    pdf_dir="/path/to/papers",
    groq_api_key="YOUR_GROQ_KEY",
    num_papers=5
)

print(f"Extracted {len(heuristics)} heuristics")
```

## Organizing Your Papers

### Directory Structure

Create a folder with your PDF papers:

```
~/research/disaster-papers/
├── housing-recovery-after-katrina.pdf
├── community-resilience-factors.pdf
├── social-networks-disaster-recovery.pdf
├── infrastructure-dependence-study.pdf
└── household-decision-making-floods.pdf
```

### File Naming

Name files descriptively - they're used for keyword filtering:

- `disaster-recovery-study.pdf` ✓
- `paper1.pdf` ✗ (won't match keywords)

## Keyword Filtering

Papers are filtered by filename keywords to find relevant ones:

### Default Keywords

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

### Custom Keywords

```python
heuristics = build_knowledge_base_from_pdfs(
    pdf_dir="/path/to/papers",
    groq_api_key="YOUR_KEY",
    keywords=['housing', 'household', 'economic'],  # Custom keywords
    num_papers=5
)
```

## Working with the PDF Retriever

### List Available Papers

```python
from household_recovery.pdf_retrieval import LocalPaperRetriever

retriever = LocalPaperRetriever("/path/to/papers")

# List all PDFs
all_papers = retriever.list_papers()
print(f"Found {len(all_papers)} PDFs")

# Filter by keywords
relevant = retriever.filter_relevant(
    keywords=['recovery', 'community'],
    max_papers=10
)
print(f"Found {len(relevant)} relevant papers")
```

### Load and Inspect Papers

```python
papers = retriever.load_papers(
    keywords=['recovery'],
    max_papers=3
)

for paper in papers:
    print(f"Title: {paper.title}")
    print(f"Abstract (first 200 chars):")
    print(f"  {paper.abstract[:200]}...")
    print()
```

## Abstract Extraction

The system automatically extracts abstracts from PDFs:

1. Looks for "Abstract" or "Summary" section markers
2. Finds text between abstract marker and introduction
3. Falls back to first ~2000 characters if no abstract found

### Manual Abstract Extraction

```python
from household_recovery.pdf_retrieval import PDFReader, extract_abstract

reader = PDFReader()
text = reader.extract_text("paper.pdf", max_pages=5)

abstract = extract_abstract(text, max_length=2000)
print(abstract)
```

## Hybrid Mode: PDFs + Scholar

Combine local papers with Google Scholar:

```python
from household_recovery.heuristics import build_knowledge_base_hybrid

heuristics = build_knowledge_base_hybrid(
    pdf_dir="/path/to/local/papers",
    serpapi_key="YOUR_SERPAPI_KEY",
    groq_api_key="YOUR_GROQ_KEY",
    scholar_query="disaster recovery agent-based model",
    num_papers=5,
    prefer_local=True  # Try local PDFs first
)
```

### Fallback Behavior

With `prefer_local=True`:
1. Try local PDFs
2. If no heuristics extracted, try Google Scholar
3. If still nothing, use fallback heuristics

## Complete Example

```python
from pathlib import Path
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.heuristics import build_knowledge_base_from_pdfs
from household_recovery.visualization import create_simulation_report

# Extract heuristics from your research library
pdf_dir = Path.home() / "research" / "disaster-papers"

heuristics = build_knowledge_base_from_pdfs(
    pdf_dir=pdf_dir,
    groq_api_key="YOUR_GROQ_KEY",
    keywords=['recovery', 'household', 'community'],
    num_papers=5
)

print(f"Extracted {len(heuristics)} heuristics:")
for h in heuristics:
    print(f"  IF {h.condition_str}")
    print(f"     THEN {h.action}")
    print(f"     Source: {h.source}")

# Run simulation with extracted heuristics
config = SimulationConfig(
    num_households=100,
    steps=25,
    network_type='watts_strogatz'
)

engine = SimulationEngine(config, heuristics=heuristics)
result = engine.run()

print(f"\nFinal recovery: {result.final_recovery:.3f}")

# Generate report
create_simulation_report(result, "./output/pdf_based")
```

## Troubleshooting

### "No PDF library found"

Install pypdf:
```bash
pip install pypdf
```

### "No text extracted"

- PDF might be image-based (scanned)
- Try a different PDF library or OCR tool
- Use a different paper

### "No heuristics extracted"

- Paper content might not be relevant
- Try different papers or keywords
- Check Groq API key is valid

### Slow Processing

- Reduce `max_pages` in PDF reading
- Use `num_papers` to limit papers processed
- Pre-extract heuristics and reuse

## Best Practices

1. **Curate your library** - Use papers specifically about disaster recovery
2. **Name files descriptively** - Helps keyword filtering
3. **Check abstracts** - Ensure papers have clear, extractable abstracts
4. **Pre-process once** - Extract heuristics once and save for reuse
5. **Combine sources** - Use hybrid mode for best coverage

## Next Steps

- [RAG Pipeline](rag-pipeline.md) - Using Google Scholar
- [Custom Parameters](custom-parameters.md) - Parameter configuration
- [Monte Carlo](monte-carlo.md) - Statistical analysis
