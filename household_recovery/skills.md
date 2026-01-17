# RecovUS Parameter Extraction Skills

This document provides the LLM prompt template and guidance for extracting RecovUS model parameters from disaster recovery research papers.

## LLM Prompt Template

Use this prompt when extracting RecovUS parameters from research papers:

```
You are extracting household disaster recovery parameters from academic research papers.
The RecovUS model simulates household recovery decisions using these parameter categories:

## PARAMETER CATEGORIES

### 1. PERCEPTION DISTRIBUTION (ASNA Index)
What factors do households prioritize when deciding whether to rebuild?
- perception_infrastructure: % who prioritize infrastructure (power, water, roads)
- perception_social: % who prioritize social networks (neighbors rebuilding)
- perception_community: % who prioritize community assets (businesses, schools, amenities)
NOTE: These three values MUST sum to 1.0 (100%)

### 2. ADEQUACY THRESHOLDS
At what recovery level do households feel the community is "adequate" for rebuilding?
- adequacy_infrastructure: Infrastructure functionality threshold (0.0-1.0)
- adequacy_neighbor: Neighbor recovery threshold (0.0-1.0)
- adequacy_community_assets: Business/amenity availability threshold (0.0-1.0)

### 3. TRANSITION PROBABILITIES
What is the probability of starting repairs under different conditions?
- transition_r0: Probability of repair when ONLY financially feasible (community not ready)
- transition_r1: Probability of repair when BOTH financially feasible AND community adequate
- transition_r2: Probability of completing repairs once started (when community adequate)
- transition_relocate: Probability of relocating when financially infeasible

### 4. FINANCIAL PARAMETERS
What are typical financial assistance rates and amounts?
- insurance_penetration_rate: % of households with insurance coverage
- fema_ha_max: Maximum FEMA Housing Assistance grant (USD)
- sba_loan_max: Maximum SBA disaster loan (USD)
- sba_uptake_rate: % of eligible households who take SBA loans
- cdbg_dr_probability: % of low-income households receiving CDBG-DR

## OUTPUT FORMAT

Return a JSON object with the following structure. Use null for parameters not found in the text.
Include exact quotes from the paper and confidence scores (0.0-1.0).

{
  "disaster_type": "<flood|hurricane|earthquake|wildfire|tornado|other>",
  "disaster_event": "<specific event name if mentioned, e.g., 'Hurricane Harvey'>",

  "perception_distribution": {
    "infrastructure": <float 0.0-1.0>,
    "social": <float 0.0-1.0>,
    "community": <float 0.0-1.0>,
    "source_quote": "<exact quote supporting this>",
    "confidence": <0.0-1.0>
  },

  "adequacy_thresholds": {
    "infrastructure": <float 0.0-1.0>,
    "neighbor": <float 0.0-1.0>,
    "community_assets": <float 0.0-1.0>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  },

  "transition_probabilities": {
    "r0": <float 0.0-1.0>,
    "r1": <float 0.0-1.0>,
    "r2": <float 0.0-1.0>,
    "relocate": <float 0.0-1.0>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  },

  "financial_parameters": {
    "insurance_penetration_rate": <float 0.0-1.0>,
    "fema_ha_average": <float USD>,
    "sba_uptake_rate": <float 0.0-1.0>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }
}

## EXTRACTION GUIDELINES

1. ONLY extract values EXPLICITLY stated or clearly implied in the text
2. Convert percentages to decimals (e.g., "65%" becomes 0.65)
3. If a timeline is given (e.g., "18 months to recovery"), compute monthly rate
4. Include EXACT quotes that support each extraction
5. Confidence scoring:
   - 0.9-1.0: Direct quote with numeric value
   - 0.7-0.9: Clear implication from text
   - 0.5-0.7: Reasonable inference
   - Below 0.5: Use null instead
6. Perception distribution MUST sum to 1.0 - adjust proportionally if needed
7. If only partial data available, extract what you can with appropriate confidence
```

---

## Parameter Definitions

### Perception Distribution (ASNA Index)

The ASNA (Awareness of Socio-spatial Network Attributes) Index categorizes households by what community factors they consider when deciding to rebuild.

| Parameter | Description | Keywords to Search | Example Quotes | Default |
|-----------|-------------|-------------------|----------------|---------|
| `perception_infrastructure` | % who wait for utilities/services | "utility", "services", "power", "water", "infrastructure", "essential services" | "65% of households waited for power restoration before beginning repairs" | 0.65 |
| `perception_social` | % who watch neighbors | "neighbors", "social", "community ties", "social networks", "peer influence" | "Households were influenced by seeing neighbors rebuild" | 0.31 |
| `perception_community` | % who wait for amenities | "businesses", "schools", "amenities", "services", "commercial", "community assets" | "Some households waited for local stores to reopen" | 0.04 |

**Extraction Tips:**
- Look for survey questions about "what factors influenced your decision to rebuild"
- Search for phrases like "most important consideration" or "primary factor"
- Often found in results sections discussing household decision-making

### Adequacy Thresholds

These define the "tipping point" at which households feel the community is recovered enough to justify rebuilding.

| Parameter | Description | Keywords to Search | Example Quotes | Default |
|-----------|-------------|-------------------|----------------|---------|
| `adequacy_infrastructure` | Infrastructure recovery level to trigger rebuilding | "functional", "restored", "operational", "threshold", "minimum level" | "Once 50% of infrastructure was restored, rebuilding accelerated" | 0.50 |
| `adequacy_neighbor` | Neighbor recovery rate to trigger rebuilding | "neighbors rebuilt", "surrounding homes", "neighborhood recovery" | "When 40% of neighbors had returned, others followed" | 0.40 |
| `adequacy_community_assets` | Business/amenity recovery to trigger rebuilding | "businesses reopened", "services available", "commercial recovery" | "Rebuilding increased after half of local businesses reopened" | 0.50 |

**Extraction Tips:**
- Look for threshold language: "once X% had...", "after reaching...", "when levels exceeded..."
- May be implicit in recovery curves or phase transitions
- Check discussion of "tipping points" or "critical mass"

### Transition Probabilities

These govern the probabilistic state machine that determines household transitions.

| Parameter | Description | Keywords to Search | Example Quotes | Default |
|-----------|-------------|-------------------|----------------|---------|
| `transition_r0` | P(repair \| feasible, ~adequate) | "despite", "even though", "before community", "early rebuilders" | "35% began repairs despite uncertain community conditions" | 0.35 |
| `transition_r1` | P(repair \| feasible, adequate) | "once conditions", "when ready", "after restoration" | "95% of financially-able households repaired when conditions were favorable" | 0.95 |
| `transition_r2` | P(complete \| started, adequate) | "completion rate", "finished repairs", "fully recovered" | "Most households who started repairs completed them within a year" | 0.95 |
| `transition_relocate` | P(relocate \| ~feasible) | "relocated", "moved away", "permanent displacement", "did not return" | "5% of households chose to relocate rather than rebuild" | 0.05 |

**Extraction Tips:**
- r0 is usually lower (20-50%): represents "pioneers" who rebuild without waiting
- r1 is usually high (85-99%): most rebuild when conditions are favorable
- Look for conditional language: "of those who could afford...", "among eligible households..."

### Financial Parameters

These configure the financial assistance landscape.

| Parameter | Description | Keywords to Search | Example Quotes | Default |
|-----------|-------------|-------------------|----------------|---------|
| `insurance_penetration_rate` | % with insurance | "insured", "coverage", "NFIP", "flood insurance" | "60% of households in the flood zone had insurance" | 0.60 |
| `fema_ha_max` | Max FEMA grant | "FEMA", "IHP", "Housing Assistance", "maximum grant" | "FEMA grants averaged $15,000, with a maximum of $35,500" | 35500.0 |
| `sba_loan_max` | Max SBA loan | "SBA", "disaster loan", "Small Business Administration" | "SBA offered loans up to $200,000 for home repairs" | 200000.0 |
| `sba_uptake_rate` | % taking SBA loans | "loan uptake", "borrowed", "accepted loans" | "40% of eligible households took SBA disaster loans" | 0.40 |
| `cdbg_dr_probability` | % receiving CDBG-DR | "CDBG", "block grant", "HUD assistance" | "30% of low-income households received CDBG-DR funds" | 0.30 |

**Extraction Tips:**
- Financial parameters are often disaster/location-specific
- Check methodology sections for sample characteristics
- FEMA limits change over time - note the year of the study

---

## Disaster-Specific Guidance

Parameters vary significantly by disaster type. Use these typical ranges as guidance.

### Flood (e.g., Hurricane Harvey, Katrina flooding)

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| perception_infrastructure | 0.60-0.75 | Infrastructure heavily impacted |
| perception_social | 0.20-0.35 | Moderate social influence |
| perception_community | 0.02-0.08 | Less focus on amenities |
| adequacy_infrastructure | 0.45-0.60 | Higher threshold needed |
| adequacy_neighbor | 0.35-0.50 | Moderate neighbor influence |
| insurance_penetration_rate | 0.50-0.80 | Higher in NFIP zones |
| transition_r0 | 0.25-0.40 | Some early rebuilders |
| transition_relocate | 0.05-0.15 | Moderate relocation |

### Hurricane (wind damage)

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| perception_infrastructure | 0.55-0.70 | Less infrastructure damage |
| perception_social | 0.25-0.40 | Higher social influence |
| perception_community | 0.03-0.10 | Some community focus |
| adequacy_infrastructure | 0.40-0.55 | Faster infrastructure recovery |
| adequacy_neighbor | 0.35-0.50 | Visual progress matters |
| insurance_penetration_rate | 0.60-0.85 | Wind coverage more common |
| transition_r1 | 0.90-0.98 | High repair rate |
| transition_relocate | 0.02-0.08 | Low relocation |

### Earthquake

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| perception_infrastructure | 0.70-0.85 | Structural concerns dominate |
| perception_social | 0.10-0.25 | Less social influence |
| perception_community | 0.02-0.08 | Low community focus |
| adequacy_infrastructure | 0.55-0.70 | Higher safety threshold |
| insurance_penetration_rate | 0.10-0.30 | Low outside CA |
| transition_r0 | 0.15-0.30 | More cautious |
| transition_relocate | 0.08-0.20 | Higher relocation |

### Wildfire

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| perception_infrastructure | 0.50-0.65 | Moderate infrastructure focus |
| perception_social | 0.20-0.35 | Community matters |
| perception_community | 0.10-0.25 | Higher - amenity loss significant |
| adequacy_community_assets | 0.50-0.70 | Businesses critical |
| insurance_penetration_rate | 0.30-0.60 | Variable coverage |
| transition_relocate | 0.10-0.25 | Higher relocation rate |

### Tornado

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| perception_infrastructure | 0.50-0.65 | Localized damage |
| perception_social | 0.30-0.45 | High social influence |
| perception_community | 0.05-0.15 | Some community focus |
| adequacy_neighbor | 0.40-0.55 | Visual progress important |
| adequacy_infrastructure | 0.35-0.50 | Lower threshold |
| transition_r0 | 0.35-0.55 | More immediate rebuilding |
| transition_r1 | 0.92-0.99 | Very high repair rate |

---

## Validation Rules

Use these rules to validate extracted parameters:

```python
RECOVUS_VALIDATION_RULES = {
    # Perception distribution must sum to 1.0
    'perception_sum': {
        'rule': lambda p: abs(p['infrastructure'] + p['social'] + p['community'] - 1.0) < 0.02,
        'message': 'Perception values must sum to 1.0'
    },

    # All thresholds in [0, 1]
    'adequacy_bounds': {
        'infrastructure': {'min': 0.2, 'max': 0.9},
        'neighbor': {'min': 0.2, 'max': 0.8},
        'community_assets': {'min': 0.2, 'max': 0.9},
    },

    # Transition probabilities
    'transition_r0': {'min': 0.10, 'max': 0.60},  # Usually 20-50%
    'transition_r1': {'min': 0.70, 'max': 1.00},  # Usually 85-99%
    'transition_r2': {'min': 0.70, 'max': 1.00},  # Usually 90-99%
    'transition_relocate': {'min': 0.01, 'max': 0.30},  # Usually 2-20%

    # r1 should be greater than r0 (community adequacy should help)
    'r1_gt_r0': {
        'rule': lambda t: t['r1'] > t['r0'],
        'message': 'r1 should be greater than r0'
    },

    # Financial parameters
    'insurance_penetration': {'min': 0.05, 'max': 0.95},
    'sba_uptake': {'min': 0.05, 'max': 0.70},
    'fema_ha_max': {'min': 20000, 'max': 50000},  # Historical range
    'sba_loan_max': {'min': 100000, 'max': 300000},  # Historical range
}
```

---

## Integration with RAG Pipeline

### Using This Prompt in Code

```python
from household_recovery.heuristics import RecovUSParameterExtractor

extractor = RecovUSParameterExtractor(
    groq_api_key="your-api-key",
    model="llama-3.3-70b-versatile"
)

# Extract from paper abstracts
params = extractor.extract(papers)

# Apply to config with confidence threshold
if params.has_any_parameters():
    recovus_config = params.apply_to_config(
        base_config,
        confidence_threshold=0.7
    )
```

### Confidence-Based Application

Parameters are only applied if their confidence score meets the threshold:

| Confidence | Interpretation | Action |
|------------|---------------|--------|
| 0.9-1.0 | Direct quote with value | Always apply |
| 0.7-0.9 | Clear implication | Apply if threshold = 0.7 |
| 0.5-0.7 | Reasonable inference | Apply if threshold = 0.5 |
| < 0.5 | Uncertain | Never apply, use default |

---

## References

- Moradi, S. & Nejat, A. (2020). RecovUS: An Agent-Based Model of Post-Disaster Household Recovery. *Journal of Artificial Societies and Social Simulation*, 23(4), 13. https://www.jasss.org/23/4/13.html
- FEMA Housing Assistance Program: https://www.fema.gov/assistance/individual/housing
- SBA Disaster Loans: https://www.sba.gov/funding-programs/disaster-assistance
