# Heuristics Module

RAG pipeline for extracting behavioral heuristics from research.

```python
from household_recovery.heuristics import (
    Heuristic,
    Paper,
    ExtractedParameters,
    ScholarRetriever,
    HeuristicExtractor,
    ParameterExtractor,
    build_knowledge_base,
    build_knowledge_base_from_pdfs,
    build_knowledge_base_hybrid,
    get_fallback_heuristics
)
```

## RAG Pattern Overview

This module implements Retrieval-Augmented Generation:
1. **RETRIEVE**: Fetch relevant academic papers from Google Scholar
2. **AUGMENT**: Use paper text as context (Scholar abstracts; PDF full-text excerpts when enabled)
3. **GENERATE**: LLM extracts actionable heuristics from research

---

## Heuristic

A behavioral rule extracted from research.

Heuristics have the form: `IF <condition> THEN <action>`

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `condition_str` | `str` | Python expression like `ctx['avg_neighbor_recovery'] > 0.5` |
| `action` | `dict[str, float]` | Action to take (utility: `{'boost': 1.5}` / `{'extra_recovery': 0.1}`; RecovUS: `{'modify_r1': 1.1}`, `{'modify_adq_nbr': -0.05}`) |
| `source` | `str` | Source of the heuristic (paper or 'fallback') |

### Methods

#### `compile() -> Heuristic`

Compile the condition string into an evaluator function.

```python
heuristic = Heuristic(
    condition_str="ctx['avg_neighbor_recovery'] > 0.5",
    action={'boost': 1.5},
    source='Social influence'
).compile()
```

#### `evaluate(ctx: dict) -> bool`

Evaluate the condition against a context dictionary.

```python
ctx = {'avg_neighbor_recovery': 0.7, 'avg_infra_func': 0.4}
if heuristic.evaluate(ctx):
    print("Condition met!")
```

---

## Paper

Represents an academic paper retrieved from Google Scholar.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Paper title |
| `abstract` | `str` | Abstract text |
| `authors` | `str` | Author names |
| `year` | `str` | Publication year |
| `link` | `str` | URL to paper |
| `cited_by` | `int` | Citation count |

---

## ExtractedParameters

Numeric parameters extracted from research papers via RAG.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `base_recovery_rate` | `float | None` | Per-step recovery rate |
| `base_recovery_rate_confidence` | `float | None` | Extraction confidence (0-1) |
| `income_threshold_low` | `float | None` | Low income threshold |
| `income_threshold_high` | `float | None` | High income threshold |
| `resilience_threshold_low` | `float | None` | Low resilience threshold |
| `resilience_threshold_high` | `float | None` | High resilience threshold |
| `utility_weight_neighbor` | `float | None` | Neighbor influence weight |
| `utility_weight_infrastructure` | `float | None` | Infrastructure weight |

### Methods

#### `to_dict() -> dict`

Convert to dictionary, excluding None values.

#### `has_any_parameters() -> bool`

Check if any parameters were extracted.

---

## ScholarRetriever

Fetches papers from Google Scholar via SerpApi.

### Constructor

```python
retriever = ScholarRetriever(
    api_key="YOUR_SERPAPI_KEY",
    cache_dir=Path(".cache/scholar"),
    cache_expiry_hours=24
)
```

### Methods

#### `search(query, num_results=5) -> list[Paper]`

Search Google Scholar for papers.

```python
papers = retriever.search(
    query="disaster recovery heuristics agent-based model",
    num_results=5
)
```

---

## HeuristicExtractor

Extracts behavioral heuristics from paper text using LLM.

### Constructor

```python
extractor = HeuristicExtractor(
    api_key="YOUR_GROQ_KEY",
    model="llama-3.3-70b-versatile",
    temperature=0.05,
    max_tokens=1200
)
```

### Methods

#### `extract(papers) -> list[Heuristic]`

Extract heuristics from paper text.

```python
heuristics = extractor.extract(papers)
for h in heuristics:
    print(f"IF {h.condition_str} THEN {h.action}")
```

---

## ParameterExtractor

Extracts numeric simulation parameters from paper text.

### Constructor

```python
extractor = ParameterExtractor(
    api_key="YOUR_GROQ_KEY",
    model="llama-3.3-70b-versatile",
    temperature=0.05,
    max_tokens=2000
)
```

### Methods

#### `extract(papers) -> ExtractedParameters`

Extract numeric parameters from papers.

```python
params = extractor.extract(papers)
if params.base_recovery_rate:
    print(f"Recovery rate: {params.base_recovery_rate}")
```

---

## Convenience Functions

### build_knowledge_base

Main entry point for the RAG pipeline.

```python
heuristics = build_knowledge_base(
    serpapi_key="YOUR_SERPAPI_KEY",
    groq_api_key="YOUR_GROQ_KEY",
    query="disaster recovery heuristics",
    num_papers=5,
    cache_dir=Path(".cache/scholar"),
    us_only=True
)
```

### build_knowledge_base_from_pdfs

Build knowledge base from local PDF files.

When `use_full_text=True` (default), the extractor receives full-text excerpts from PDFs; set it to `False` to use extracted abstracts only.

```python
heuristics = build_knowledge_base_from_pdfs(
    pdf_dir=Path("~/research/papers"),
    groq_api_key="YOUR_GROQ_KEY",
    keywords=['recovery', 'disaster'],
    num_papers=5,
    us_only=True,
    use_full_text=True,
    pdf_max_pages=None
)
```

### build_knowledge_base_hybrid

Combine local PDFs and Google Scholar.

```python
heuristics = build_knowledge_base_hybrid(
    pdf_dir=Path("~/research/papers"),
    serpapi_key="YOUR_SERPAPI_KEY",
    groq_api_key="YOUR_GROQ_KEY",
    scholar_query="disaster recovery",
    num_papers=5,
    prefer_local=True,
    us_only=True,
    use_full_text=True,
    pdf_max_pages=None
)
```

### get_fallback_heuristics

Return static fallback heuristics when LLM extraction fails.

```python
heuristics = get_fallback_heuristics()
# Returns 6 default heuristics based on common research findings
```

**Default Fallback Heuristics:**
1. Neighbor influence: boost if avg_neighbor_recovery > 0.5
2. Infrastructure barriers: reduce if avg_infra_func < 0.3
3. Economic incentive: boost if avg_business_avail > 0.6
4. Network cohesion: extra recovery if num_neighbors > 4
5. High resilience: boost if resilience > 0.7
6. Vulnerability compound: reduce if low income AND poor infrastructure

---

## Allowed Context Keys

Heuristic conditions can only reference these context keys:

```python
ALLOWED_CTX_KEYS = {
    'avg_neighbor_recovery',    # Average neighbor recovery (0-1)
    'avg_infra_func',           # Average infrastructure functionality (0-1)
    'avg_business_avail',       # Average business availability (0-1)
    'num_neighbors',            # Number of neighbors (int)
    'resilience',               # Household resilience (0-1)
    'resilience_category',      # 'low', 'medium', 'high'
    'household_income',         # Annual income (float)
    'income_level',             # 'low', 'middle', 'high'
    'perception_type',          # RecovUS perception type
    'damage_severity',          # RecovUS damage severity
    'recovery_state',           # RecovUS recovery state
    'is_feasible',              # Financial feasibility
    'is_adequate',              # Community adequacy
    'is_habitable',             # Habitability flag
    'repair_cost',              # Estimated repair cost
    'available_resources',      # Total available resources
    'time_step',                # Current simulation step
    'months_since_disaster',    # Months since disaster
    'avg_neighbor_recovered_binary',  # % of neighbors fully recovered
}
```

---

## Parameter Validation Rules

Extracted parameters are validated against bounds:

| Parameter | Min | Max | Default |
|-----------|-----|-----|---------|
| `base_recovery_rate` | 0.01 | 0.5 | 0.1 |
| `income_threshold_low` | 10000 | 100000 | 45000 |
| `income_threshold_high` | 50000 | 500000 | 120000 |
| `resilience_threshold_low` | 0.1 | 0.5 | 0.35 |
| `resilience_threshold_high` | 0.5 | 0.95 | 0.70 |
| `utility_weight_neighbor` | 0.0 | 1.0 | 0.3 |
| `utility_weight_infrastructure` | 0.0 | 1.0 | 0.2 |
