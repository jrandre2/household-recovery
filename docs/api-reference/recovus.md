# RecovUS Module

The RecovUS module implements the Moradi & Nejat (2020) decision model for sophisticated household disaster recovery simulation.

## Overview

RecovUS models household recovery decisions using:

1. **Perception Types (ASNA Index)**: What community factors households prioritize
2. **Financial Feasibility**: Whether households can afford repairs
3. **Community Adequacy**: Whether the community is recovered enough
4. **State Machine**: Probabilistic transitions between recovery states

## RecovUSConfig

Configuration dataclass for all RecovUS parameters.

```python
from household_recovery.config import RecovUSConfig
```

### Constructor

```python
RecovUSConfig(
    enabled: bool = True,
    # Perception distribution (must sum to 1.0)
    perception_infrastructure: float = 0.65,
    perception_social: float = 0.31,
    perception_community: float = 0.04,
    # Adequacy thresholds
    adequacy_infrastructure: float = 0.50,
    adequacy_neighbor: float = 0.40,
    adequacy_community_assets: float = 0.50,
    # Transition probabilities
    transition_r0: float = 0.35,
    transition_r1: float = 0.95,
    transition_r2: float = 0.95,
    transition_relocate: float = 0.05,
    # Financial parameters
    insurance_penetration_rate: float = 0.60,
    fema_ha_max: float = 35500.0,
    sba_loan_max: float = 200000.0,
    sba_income_floor: float = 30000.0,
    sba_uptake_rate: float = 0.40,
    cdbg_dr_coverage_rate: float = 0.50,
    cdbg_dr_probability: float = 0.30,
    # Damage costs
    damage_cost_minor: float = 0.10,
    damage_cost_moderate: float = 0.30,
    damage_cost_severe: float = 0.60,
    damage_cost_destroyed: float = 1.00,
    temp_housing_monthly: float = 1500.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable/disable RecovUS model |
| `perception_infrastructure` | float | 0.65 | % of households watching infrastructure |
| `perception_social` | float | 0.31 | % of households watching neighbors |
| `perception_community` | float | 0.04 | % of households watching businesses |
| `adequacy_infrastructure` | float | 0.50 | Infrastructure threshold for adequacy |
| `adequacy_neighbor` | float | 0.40 | Neighbor recovery threshold |
| `adequacy_community_assets` | float | 0.50 | Business availability threshold |
| `transition_r0` | float | 0.35 | P(repair \| feasible, ~adequate) |
| `transition_r1` | float | 0.95 | P(repair \| feasible, adequate) |
| `transition_r2` | float | 0.95 | P(complete \| repairing, adequate) |
| `transition_relocate` | float | 0.05 | P(relocate \| ~feasible) |
| `insurance_penetration_rate` | float | 0.60 | % with insurance coverage |
| `fema_ha_max` | float | 35500.0 | Max FEMA Housing Assistance ($) |
| `sba_loan_max` | float | 200000.0 | Max SBA disaster loan ($) |
| `sba_income_floor` | float | 30000.0 | Min income for SBA eligibility |
| `sba_uptake_rate` | float | 0.40 | % taking SBA loans if eligible |
| `cdbg_dr_coverage_rate` | float | 0.50 | CDBG-DR coverage fraction |
| `cdbg_dr_probability` | float | 0.30 | % receiving CDBG-DR |

### Methods

#### validate()

Validates configuration parameters.

```python
config = RecovUSConfig()
config.validate()  # Raises ValueError if invalid
```

#### get_transition_probabilities()

Returns transition probabilities as dictionary.

```python
probs = config.get_transition_probabilities()
# {'r0': 0.35, 'r1': 0.95, 'r2': 0.95, 'relocate': 0.05}
```

#### get_adequacy_thresholds()

Returns adequacy thresholds as dictionary.

```python
thresholds = config.get_adequacy_thresholds()
# {'infrastructure': 0.50, 'neighbor': 0.40, 'community_assets': 0.50}
```

---

## RecoveryStateMachine

Manages probabilistic state transitions for household recovery.

```python
from household_recovery.recovus import RecoveryStateMachine, TransitionProbabilities
```

### Constructor

```python
RecoveryStateMachine(
    probabilities: TransitionProbabilities | None = None,
    rng: np.random.Generator | None = None,
)
```

### States

| State | Description |
|-------|-------------|
| `'waiting'` | Household waiting to start repairs |
| `'repairing'` | Actively repairing home |
| `'recovered'` | Repairs complete, household stable |
| `'relocated'` | Permanently left community |

### Methods

#### transition()

Determine state transition for a household.

```python
new_state, action, increment = machine.transition(
    current_state='waiting',
    current_recovery=0.0,
    is_feasible=True,
    is_adequate=True,
    base_repair_rate=0.1,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_state` | str | Current recovery state |
| `current_recovery` | float | Current recovery level (0-1) |
| `is_feasible` | bool | Financial feasibility |
| `is_adequate` | bool | Community adequacy |
| `base_repair_rate` | float | Base recovery increment per step |

**Returns:** `tuple[str, str, float]`
- New state
- Action taken
- Recovery increment

#### reset_rng()

Reset the random generator.

```python
machine.reset_rng(seed=42)
```

---

## TransitionProbabilities

Dataclass for state transition probabilities.

```python
from household_recovery.recovus import TransitionProbabilities
```

### Constructor

```python
TransitionProbabilities(
    r0: float = 0.35,
    r1: float = 0.95,
    r2: float = 0.95,
    relocate_when_infeasible: float = 0.05,
    relocate_when_inadequate: float = 0.02,
)
```

### Methods

#### copy()

Create a copy of the probabilities.

```python
new_probs = probs.copy()
```

#### to_dict()

Convert to dictionary.

```python
d = probs.to_dict()
# {'r0': 0.35, 'r1': 0.95, 'r2': 0.95, ...}
```

---

## RecovUSExtractedParameters

Structured container for RecovUS parameters extracted from research papers.

```python
from household_recovery import RecovUSExtractedParameters
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `disaster_type` | str \| None | Type of disaster studied |
| `disaster_event` | str \| None | Specific event name |
| `perception_distribution` | dict \| None | Extracted perception percentages |
| `adequacy_thresholds` | dict \| None | Extracted adequacy thresholds |
| `transition_probabilities` | dict \| None | Extracted transition probs |
| `financial_parameters` | dict \| None | Extracted financial params |

### Methods

#### has_any_parameters()

Check if any parameters were extracted.

```python
if params.has_any_parameters():
    # Apply to config
```

#### apply_to_config()

Apply extracted parameters to a base config.

```python
base_config = RecovUSConfig()
updated_config = params.apply_to_config(
    base_config,
    confidence_threshold=0.7  # Only apply if confidence >= 0.7
)
```

#### summary()

Get human-readable summary of extracted parameters.

```python
print(params.summary())
```

---

## RecovUSParameterExtractor

LLM-based extractor for RecovUS parameters from research papers.

```python
from household_recovery import RecovUSParameterExtractor
```

### Constructor

```python
RecovUSParameterExtractor(
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.05,
)
```

### Methods

#### extract()

Extract RecovUS parameters from papers.

```python
extractor = RecovUSParameterExtractor(groq_api_key="...")
params = extractor.extract(papers)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `papers` | list[Paper] | Papers to extract from |

**Returns:** `RecovUSExtractedParameters`

---

## KnowledgeBaseResult

Combined result from full knowledge base extraction.

```python
from household_recovery import KnowledgeBaseResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `heuristics` | list[dict] | Extracted behavioral heuristics |
| `recovus_params` | RecovUSExtractedParameters \| None | RecovUS parameters |
| `papers_processed` | int | Number of papers processed |
| `extraction_time` | float | Extraction duration (seconds) |

### Methods

#### has_recovus_params()

Check if RecovUS parameters were extracted.

```python
if result.has_recovus_params():
    config = result.recovus_params.apply_to_config(base)
```

#### summary()

Get human-readable summary.

```python
print(result.summary())
# Papers processed: 5
# Heuristics extracted: 8
# RecovUS parameters: perception, adequacy, transition
```

---

## Helper Functions

### build_full_knowledge_base()

Build knowledge base with both heuristics and RecovUS extraction.

```python
from household_recovery import build_full_knowledge_base

result = build_full_knowledge_base(
    serpapi_key="your-key",
    groq_api_key="your-key",
    query="household disaster recovery",
    num_papers=5,
    extract_recovus=True,
)
```

### build_full_knowledge_base_from_pdfs()

Build from local PDF files.

```python
from household_recovery import build_full_knowledge_base_from_pdfs

result = build_full_knowledge_base_from_pdfs(
    pdf_dir=Path("~/papers"),
    groq_api_key="your-key",
    extract_recovus=True,
)
```

### build_full_knowledge_base_hybrid()

Build from multiple sources.

```python
from household_recovery import build_full_knowledge_base_hybrid

result = build_full_knowledge_base_hybrid(
    serpapi_key="your-key",
    groq_api_key="your-key",
    pdf_dir=Path("~/papers"),
    query="disaster recovery",
    extract_recovus=True,
)
```

### get_recovus_fallback_heuristics()

Get default RecovUS-aware heuristics.

```python
from household_recovery.heuristics import get_recovus_fallback_heuristics

heuristics = get_recovus_fallback_heuristics()
```

---

## Type Aliases

```python
from household_recovery.recovus import RecoveryState, RecoveryAction, PerceptionType

RecoveryState = Literal['waiting', 'repairing', 'recovered', 'relocated']
RecoveryAction = Literal['none', 'wait', 'start_repair', 'repair_progress',
                         'repair_slow', 'complete', 'sell_relocate']
PerceptionType = Literal['infrastructure', 'social', 'community']
```

---

## References

- Moradi, S. & Nejat, A. (2020). RecovUS: An Agent-Based Model of Post-Disaster Household Recovery. *Journal of Artificial Societies and Social Simulation*, 23(4), 13. https://www.jasss.org/23/4/13.html

## See Also

- [RecovUS User Guide](../user-guide/recovus.md)
- [State Diagrams](../architecture/state-diagrams.md)
- [Configuration Reference](config.md)
