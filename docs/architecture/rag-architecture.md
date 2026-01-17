# RAG Architecture

Deep dive into the Retrieval-Augmented Generation pipeline.

## Overview

The RAG pipeline grounds simulation behavior in academic research by:

1. **Retrieving** relevant papers from Google Scholar or local PDFs
2. **Augmenting** an LLM prompt with paper text (Scholar abstracts; PDF full-text excerpts when enabled)
3. **Generating** structured heuristics from the research

## Pipeline Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
├──────────────────────────┬──────────────────────────────────────┤
│     Google Scholar       │         Local PDFs                    │
│     (via SerpAPI)        │         (via pypdf)                   │
└──────────────────────────┴──────────────────────────────────────┘
              │                            │
              ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   ScholarRetriever       │  │   LocalPaperRetriever    │
│   • Query Scholar        │  │   • Scan directory       │
│   • Parse results        │  │   • Filter by keywords   │
│   • Cache responses      │  │   • Extract text         │
└──────────────────────────┘  └──────────────────────────┘
              │                            │
              └────────────┬───────────────┘
                           ▼
                    [Paper objects]
                           │
                           ▼
              ┌────────────────────────┐
              │  HeuristicExtractor    │
              │  • Format prompt       │
              │  • Call LLM            │
              │  • Parse response      │
              │  • Validate conditions │
              │  • Compile heuristics  │
              └────────────────────────┘
                           │
                           ▼
                    [Heuristic objects]
```

## Retrieval Strategies

### Google Scholar (ScholarRetriever)

**Advantages:**
- Access to vast academic literature
- Citation counts for quality signals
- Automatic relevance ranking

**Implementation:**
```python
class ScholarRetriever:
    def search(self, query: str, num_results: int = 5) -> list[Paper]:
        # 1. Check cache
        if cached := self._load_from_cache(query):
            return cached

        # 2. Call SerpAPI
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        results = GoogleSearch(params).get_dict()

        # 3. Parse into Paper objects
        papers = [Paper(...) for r in results["organic_results"]]

        # 4. Cache and return
        self._save_to_cache(query, papers)
        return papers
```

### Local PDFs (LocalPaperRetriever)

**Advantages:**
- Works offline
- Use specific papers you trust
- No API costs

**Implementation:**
```python
class LocalPaperRetriever:
    def load_papers(self, keywords: list[str], max_papers: int,
                    max_pages: int | None, us_only: bool) -> list[LocalPaper]:
        # 1. Find matching PDFs
        pdf_paths = self.filter_relevant(keywords, max_papers)

        # 2. Extract text from each
        papers = []
        for path in pdf_paths:
            text = self.reader.extract_text(path, max_pages=max_pages)
            abstract = extract_abstract(text)
            if us_only and not is_us_based(text):
                continue
            papers.append(LocalPaper(title=path.stem, abstract=abstract, full_text=text, ...))

        return papers
```

### Hybrid Mode

```python
def build_knowledge_base_hybrid(pdf_dir, serpapi_key, groq_key, prefer_local=True):
    if prefer_local and pdf_dir:
        heuristics = build_knowledge_base_from_pdfs(pdf_dir, groq_key)
        if heuristics and not all_fallback(heuristics):
            return heuristics

    if serpapi_key:
        heuristics = build_knowledge_base(serpapi_key, groq_key)
        if heuristics and not all_fallback(heuristics):
            return heuristics

    return get_fallback_heuristics()
```

## LLM Extraction

### Heuristic Extraction Prompt

```python
PROMPT_TEMPLATE = """You are a strict JSON generator. Return ONLY a valid JSON array.

Task: Extract 4–6 actionable heuristics for utility-based household agents
in disaster recovery simulations.

Rules:
- condition: valid Python expression using ONLY these ctx keys:
  'avg_neighbor_recovery', 'avg_infra_func', 'avg_business_avail',
  'num_neighbors', 'resilience', 'resilience_category',
  'household_income', 'income_level'
- action: dict like {{"boost": 1.5}} or {{"extra_recovery": 0.1}}
- source: very short string

Output format:
[
  {{"condition": "ctx['avg_neighbor_recovery'] > 0.5",
    "action": {{"boost": 1.5}},
    "source": "social influence"}},
  ...
]

Text excerpts:
{abstracts}"""
```

### Key Design Decisions

1. **Strict JSON output** - Easier to parse, fewer errors
2. **Explicit allowed keys** - Prevents hallucinated variables
3. **Low temperature (0.05)** - Deterministic, reproducible output
4. **Structured output mode** - When available, use LLM's JSON mode

### Parameter Extraction

Separate prompt for numeric parameters:

```python
PARAM_PROMPT = """Extract NUMERIC PARAMETERS mentioned in the research:
- base_recovery_rate: float 0.0-1.0
- income_thresholds: {low, high}
- resilience_thresholds: {low, high}
- social_influence_weight: float
- infrastructure_weight: float

Include confidence scores (0.0-1.0) for each extraction.
Only extract values EXPLICITLY mentioned in the text."""
```

## Validation Pipeline

### Condition Validation

```python
def _compile_heuristics(self, raw_heuristics: list[dict]) -> list[Heuristic]:
    compiled = []
    for h in raw_heuristics:
        # 1. Check required keys
        if not {'condition', 'action', 'source'}.issubset(h.keys()):
            continue

        # 2. Validate condition with safe_eval
        is_valid, error = validate_condition(h['condition'])
        if not is_valid:
            logger.warning(f"Invalid condition: {error}")
            continue

        # 3. Compile into callable
        heuristic = Heuristic(
            condition_str=h['condition'],
            action=h['action'],
            source=h['source']
        ).compile()

        compiled.append(heuristic)

    return compiled
```

### Parameter Validation

```python
PARAMETER_VALIDATION_RULES = {
    'base_recovery_rate': {'min': 0.01, 'max': 0.5, 'default': 0.1},
    'income_threshold_low': {'min': 10000, 'max': 100000, 'default': 45000},
    # ...
}

def _validate_param(self, name: str, value: float) -> float | None:
    rules = PARAMETER_VALIDATION_RULES.get(name)
    if value < rules['min'] or value > rules['max']:
        logger.warning(f"{name}={value} out of bounds, rejecting")
        return None
    return value
```

## Caching Strategy

### Cache Structure

```
.cache/scholar/
├── a1b2c3d4e5f6.json  # MD5 hash of query
├── f6e5d4c3b2a1.json
└── ...
```

### Cache Entry

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "query": "disaster recovery heuristics",
  "papers": [
    {
      "title": "Paper Title",
      "abstract": "...",
      "authors": "Author et al.",
      "year": "2023",
      "link": "https://...",
      "cited_by": 42
    }
  ]
}
```

### Cache Expiry

```python
if datetime.now() - cached_time > timedelta(hours=cache_expiry_hours):
    return None  # Expired, fetch fresh
```

## Fallback Mechanism

When RAG fails, use built-in heuristics:

```python
def get_fallback_heuristics() -> list[Heuristic]:
    return [
        # Social influence
        Heuristic(
            condition_str="ctx['avg_neighbor_recovery'] > 0.5",
            action={'boost': 1.5},
            source='Neighbor influence'
        ).compile(),

        # Infrastructure barriers
        Heuristic(
            condition_str="ctx['avg_infra_func'] < 0.3",
            action={'boost': 0.6},
            source='Infrastructure barriers'
        ).compile(),

        # ... more fallbacks
    ]
```

## Error Handling

### LLM Failures

```python
try:
    response = structured_llm.invoke(prompt)
except Exception as e:
    logger.warning(f"Structured mode failed: {e}")
    # Fallback to raw call
    raw_response = llm.invoke(prompt).content
    return self._parse_raw_response(raw_response)
```

### Parsing Failures

```python
def _parse_raw_response(self, raw: str) -> list[dict]:
    # Remove markdown fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]

    # Find JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []  # Will trigger fallback
```

## Next Steps

- [Agent Model](agent-model.md) - How heuristics are applied
- [Security](security.md) - Safe condition evaluation
- [Data Flow](data-flow.md) - Overall pipeline flow
