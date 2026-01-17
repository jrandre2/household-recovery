# RAG Pipeline Guide

Learn how to use the Retrieval-Augmented Generation pipeline to extract behavioral heuristics from academic research.

## What is RAG?

RAG (Retrieval-Augmented Generation) grounds the simulation in actual research:

1. **Retrieve**: Fetch academic papers from Google Scholar
2. **Augment**: Use paper text as context (Scholar abstracts; PDF full-text excerpts when enabled)
3. **Generate**: LLM extracts actionable heuristics

This ensures simulation behavior is based on real research findings rather than arbitrary assumptions.

## Setting Up API Keys

### SerpAPI (Google Scholar)

1. Sign up at [serpapi.com](https://serpapi.com)
2. Get your API key from the dashboard
3. Set environment variable:

```bash
export SERPAPI_KEY=your_key_here
```

### Groq (LLM)

1. Sign up at [groq.com](https://groq.com)
2. Generate an API key
3. Set environment variable:

```bash
export GROQ_API_KEY=your_key_here
```

### Using .env File

Create `.env` in project root:

```
SERPAPI_KEY=your_serpapi_key
GROQ_API_KEY=your_groq_key
```

## Running with RAG

### Command Line

```bash
# With environment variables set
python -m household_recovery --households 50 --steps 20

# With explicit keys
python -m household_recovery \
    --serpapi-key YOUR_SERPAPI_KEY \
    --groq-key YOUR_GROQ_KEY \
    --households 50
```

### Python API

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import APIConfig

config = SimulationConfig(num_households=50, steps=20)
api_config = APIConfig(
    serpapi_key="YOUR_KEY",
    groq_api_key="YOUR_KEY"
)

engine = SimulationEngine(config, api_config=api_config)
result = engine.run()
```

## Understanding Extracted Heuristics

### Heuristic Format

Heuristics have the form `IF <condition> THEN <action>`:

```
IF ctx['avg_neighbor_recovery'] > 0.5 THEN {'boost': 1.5}
IF ctx['avg_infra_func'] < 0.3 THEN {'boost': 0.6}
IF ctx['resilience'] > 0.7 THEN {'extra_recovery': 0.1}
```

### Actions

- `{'boost': 1.5}` - Multiply recovery rate by 1.5
- `{'boost': 0.6}` - Reduce recovery rate to 60%
- `{'extra_recovery': 0.1}` - Add 0.1 to recovery increment

### RecovUS Actions

When RecovUS is enabled, heuristics modify transition probabilities and adequacy thresholds:

- `{'modify_r0': 1.1}` - Multiply r0 (repair when feasible only)
- `{'modify_r1': 0.9}` - Multiply r1 (repair when feasible + adequate)
- `{'modify_r2': 1.05}` - Multiply r2 (complete when repairing)
- `{'modify_adq_infr': -0.05}` - Lower infrastructure adequacy threshold
- `{'modify_adq_nbr': 0.05}` - Raise neighbor adequacy threshold
- `{'modify_adq_cas': -0.1}` - Lower community assets threshold

### Context Variables

Heuristics can reference these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `avg_neighbor_recovery` | float | Average neighbor recovery (0-1) |
| `avg_infra_func` | float | Infrastructure functionality (0-1) |
| `avg_business_avail` | float | Business availability (0-1) |
| `num_neighbors` | int | Number of neighbors |
| `resilience` | float | Household resilience (0-1) |
| `resilience_category` | str | 'low', 'medium', 'high' |
| `household_income` | float | Annual income |
| `income_level` | str | 'low', 'middle', 'high' |

### RecovUS Context Variables

| Variable | Type | Description |
|----------|------|-------------|
| `perception_type` | str | 'infrastructure', 'social', 'community' |
| `damage_severity` | str | 'none', 'minor', 'moderate', 'severe', 'destroyed' |
| `recovery_state` | str | 'waiting', 'repairing', 'recovered', 'relocated' |
| `is_feasible` | bool | Financial feasibility flag |
| `is_adequate` | bool | Community adequacy flag |
| `is_habitable` | bool | Whether home is habitable |
| `repair_cost` | float | Estimated repair cost |
| `available_resources` | float | Total available resources |
| `time_step` | int | Current simulation step |
| `months_since_disaster` | int | Months since disaster |
| `avg_neighbor_recovered_binary` | float | % of neighbors fully recovered |

## Extracted Parameters

Beyond heuristics, the RAG pipeline extracts numeric parameters:

- **Base recovery rate** from disaster timelines
- **Income thresholds** from socioeconomic research
- **Resilience thresholds** from empirical studies
- **Utility weights** from network analysis
- **RecovUS parameters** (perception, adequacy, transitions, financials) when RecovUS extraction is enabled

### Parameter Precedence

1. **RAG-extracted** (if confidence >= 0.7)
2. **Config file** values
3. **Hardcoded defaults**

## Customizing the Search

### Change Search Query

```python
from household_recovery.config import ResearchConfig

research_config = ResearchConfig(
    default_query="household disaster recovery social networks",
    num_papers=7
)

engine = SimulationEngine(
    config,
    api_config=api_config,
    research_config=research_config
)
```

### US-Only and PDF Settings

```python
from household_recovery.config import ResearchConfig

research_config = ResearchConfig(
    us_only=True,          # Filter to US-based studies
    pdf_use_full_text=True,  # Use full PDF text for local PDFs
    pdf_max_pages=None     # None = all pages
)
```

### Using Cached Results

Papers are cached to avoid repeated API calls:

```python
research_config = ResearchConfig(
    cache_dir=Path(".cache/scholar"),
    cache_expiry_hours=48  # Use cache for 2 days
)
```

## Viewing Extracted Heuristics

```python
# Build knowledge base directly
from household_recovery.heuristics import build_knowledge_base

heuristics = build_knowledge_base(
    serpapi_key="YOUR_KEY",
    groq_api_key="YOUR_KEY",
    query="disaster recovery heuristics",
    num_papers=5
)

for h in heuristics:
    print(f"IF {h.condition_str}")
    print(f"   THEN {h.action}")
    print(f"   Source: {h.source}")
    print()
```

## Fallback Behavior

If API keys are not set or extraction fails, the system uses fallback heuristics:

```python
from household_recovery.heuristics import get_fallback_heuristics

fallbacks = get_fallback_heuristics()
# Returns 6 default heuristics based on common research findings
```

### Default Fallback Heuristics

1. **Neighbor influence**: Boost if neighbors mostly recovered
2. **Infrastructure barriers**: Reduce if infrastructure poor
3. **Economic incentive**: Boost if businesses available
4. **Network cohesion**: Extra recovery if well-connected
5. **High resilience**: Boost if resilient household
6. **Vulnerability compound**: Reduce if low income AND poor infrastructure

## Pre-Building Heuristics

For reproducibility, extract heuristics once and reuse:

```python
# Extract once
heuristics = build_knowledge_base(...)

# Save for later (serialize)
import pickle
with open("heuristics.pkl", "wb") as f:
    pickle.dump([{
        'condition_str': h.condition_str,
        'action': h.action,
        'source': h.source
    } for h in heuristics], f)

# Load and reuse
with open("heuristics.pkl", "rb") as f:
    data = pickle.load(f)

from household_recovery.heuristics import Heuristic
heuristics = [
    Heuristic(**d).compile() for d in data
]

engine = SimulationEngine(config, heuristics=heuristics)
```

## Troubleshooting

### "No papers found"

- Check your SerpAPI key is valid
- Try a different search query
- Check your API quota

### "Heuristic extraction failed"

- Check your Groq API key
- Papers might not contain actionable heuristics
- System will use fallback heuristics

### Low Confidence Parameters

Parameters with confidence < 0.7 are not used. This threshold can be adjusted:

```python
from household_recovery.simulation import ParameterMerger

merger = ParameterMerger(
    sim_config=config,
    thresholds=thresholds,
    extracted=extracted_params,
    confidence_threshold=0.5  # Lower threshold
)
```

## Next Steps

- [Local PDFs](local-pdfs.md) - Use your own research library
- [Custom Parameters](custom-parameters.md) - Override RAG values
- [Monte Carlo](monte-carlo.md) - Test heuristic effects
