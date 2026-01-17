# RecovUS Decision Model

This guide explains how to use the RecovUS decision model for sophisticated household disaster recovery simulation.

## What is RecovUS?

RecovUS is an advanced household recovery decision model based on the research of Moradi & Nejat (2020). Unlike the simpler utility-based model, RecovUS captures the complex interplay of:

1. **Perception Types**: How different households perceive community recovery
2. **Financial Feasibility**: Whether households can afford to repair
3. **Community Adequacy**: Whether the community has recovered enough to warrant rebuilding

This creates a more realistic simulation of household recovery decisions after disasters.

## Quick Start

### Enabling RecovUS

RecovUS is **enabled by default** in version 0.2.0+. To explicitly enable or disable:

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import RecovUSConfig

# Enable RecovUS (default)
recovus_config = RecovUSConfig(enabled=True)

# Or disable to use utility-based model
recovus_config = RecovUSConfig(enabled=False)

config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config, recovus_config=recovus_config)
result = engine.run()
```

### Command Line

```bash
# RecovUS is enabled by default
python -m household_recovery --households 50 --steps 20

# The simulation will log RecovUS status:
# INFO: RecovUS enabled: perception distribution = (65% infra, 31% social, 4% community)
```

## Understanding Perception Types (ASNA Index)

The ASNA (Awareness of Socio-spatial Network Attributes) Index classifies households by what community factors they prioritize when deciding to rebuild.

### The Three Perception Types

| Type | Default % | What They Watch | Description |
|------|-----------|-----------------|-------------|
| **Infrastructure** | 65% | Power, water, roads | Wait for essential services to be restored |
| **Social** | 31% | Neighbors rebuilding | Influenced by seeing neighbors return |
| **Community** | 4% | Businesses, schools | Wait for amenities to reopen |

### How Perception Affects Decisions

Each perception type has its own "adequacy threshold" - the recovery level at which a household considers the community ready for rebuilding:

```python
# Default thresholds
recovus_config = RecovUSConfig(
    # Perception distribution
    perception_infrastructure=0.65,  # 65% infrastructure-aware
    perception_social=0.31,          # 31% social-aware
    perception_community=0.04,       # 4% community-aware

    # Adequacy thresholds
    adequacy_infrastructure=0.50,    # Infrastructure must be 50% functional
    adequacy_neighbor=0.40,          # 40% of neighbors must be recovered
    adequacy_community_assets=0.50,  # 50% of businesses must be open
)
```

### Customizing Perception Distribution

Adjust based on your disaster type or community characteristics:

```python
# Hurricane-affected coastal community (more social ties)
coastal_config = RecovUSConfig(
    perception_infrastructure=0.55,
    perception_social=0.40,
    perception_community=0.05,
)

# Urban area (more infrastructure-dependent)
urban_config = RecovUSConfig(
    perception_infrastructure=0.75,
    perception_social=0.20,
    perception_community=0.05,
)
```

## Financial Feasibility Model

RecovUS determines if a household can afford repairs using a 5-resource model.

### Financial Resources

| Resource | Description | Typical Amounts |
|----------|-------------|-----------------|
| **Insurance** | Homeowner's or flood insurance payout | Up to damage amount |
| **FEMA-HA** | FEMA Housing Assistance grants | Max $35,500 |
| **SBA Loans** | Small Business Administration disaster loans | Max $200,000 |
| **Liquid Assets** | Savings, accessible funds | 1-20% of net worth |
| **CDBG-DR** | Community Development Block Grants | 50% of costs for low-income |

### Feasibility Calculation

A household is **financially feasible** when:

```
Total Resources >= Repair Cost + Temporary Housing Costs
```

### Configuring Financial Parameters

```python
recovus_config = RecovUSConfig(
    # Insurance
    insurance_penetration_rate=0.60,  # 60% have insurance

    # FEMA
    fema_ha_max=35500.0,  # Maximum FEMA grant

    # SBA Loans
    sba_loan_max=200000.0,     # Maximum SBA loan
    sba_income_floor=30000.0,  # Minimum income for eligibility
    sba_uptake_rate=0.40,      # 40% of eligible take loans

    # CDBG-DR (for low-income households)
    cdbg_dr_probability=0.30,     # 30% receive CDBG-DR
    cdbg_dr_coverage_rate=0.50,   # Covers 50% of costs

    # Damage costs (as fraction of home value)
    damage_cost_minor=0.10,
    damage_cost_moderate=0.30,
    damage_cost_severe=0.60,
    damage_cost_destroyed=1.00,
)
```

## State Machine Transitions

RecovUS uses a probabilistic state machine to model recovery decisions.

### Recovery States

```
    ┌─────────┐     feasible + adequate     ┌───────────┐
    │         │ ────────── r1 ────────────> │           │
    │ WAITING │                              │ REPAIRING │
    │         │ <──────── (rare) ─────────── │           │
    └────┬────┘                              └─────┬─────┘
         │                                         │
         │ feasible only                           │ adequate
         │ (r0, lower prob)                        │ (r2)
         │                                         │
         │                                         v
         │                                   ┌───────────┐
         │                                   │ RECOVERED │
         │                                   └───────────┘
         │
         │ not feasible
         │ (relocate prob)
         v
    ┌───────────┐
    │ RELOCATED │
    └───────────┘
```

### Transition Probabilities

| Transition | Parameter | Default | Description |
|------------|-----------|---------|-------------|
| waiting → repairing (only feasible) | `r0` | 35% | Repair despite community not ready |
| waiting → repairing (feasible + adequate) | `r1` | 95% | Repair when conditions are favorable |
| repairing → recovered | `r2` | 95% | Complete repairs when adequate |
| waiting → relocated | `relocate` | 5% | Leave when financially infeasible |

### Configuring Transition Probabilities

```python
# More cautious community (lower r0, higher r1)
cautious_config = RecovUSConfig(
    transition_r0=0.20,      # Only 20% rebuild early
    transition_r1=0.98,      # 98% rebuild when ready
    transition_r2=0.95,
    transition_relocate=0.05,
)

# More resilient community (higher r0)
resilient_config = RecovUSConfig(
    transition_r0=0.50,      # 50% rebuild early
    transition_r1=0.95,
    transition_r2=0.95,
    transition_relocate=0.03,  # Lower relocation
)
```

## RAG Parameter Extraction

The RecovUS model can be parameterized from research papers using RAG extraction.

### Automatic Extraction

```python
from household_recovery import (
    build_full_knowledge_base,
    SimulationEngine,
    SimulationConfig,
)
from household_recovery.config import RecovUSConfig

# Extract parameters from Google Scholar
result = build_full_knowledge_base(
    serpapi_key="your-serpapi-key",
    groq_api_key="your-groq-key",
    query="household disaster recovery decision making",
    extract_recovus=True,  # Enable RecovUS extraction
)

# Check what was extracted
print(result.summary())
# Papers processed: 5
# Heuristics extracted: 8
# RecovUS parameters extracted:
#   - Perception: infra=70%, social=25%, community=5%
#   - Adequacy: infra=55%, neighbor=45%, community=50%

# Apply to configuration
base_config = RecovUSConfig()
if result.has_recovus_params():
    recovus_config = result.recovus_params.apply_to_config(
        base_config,
        confidence_threshold=0.7
    )
else:
    recovus_config = base_config

# Run simulation with extracted parameters
config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config, recovus_config=recovus_config)
simulation_result = engine.run()
```

### Extracted Parameters

The RAG pipeline can extract:

- **Perception distribution**: Infrastructure/social/community percentages
- **Adequacy thresholds**: Recovery levels that trigger rebuilding
- **Transition probabilities**: r0, r1, r2 values
- **Financial parameters**: Insurance rates, FEMA averages, SBA uptake

### Confidence Thresholds

Parameters are only applied if extracted with sufficient confidence:

| Confidence | Interpretation | Applied? |
|------------|----------------|----------|
| 0.9-1.0 | Direct quote with value | Yes |
| 0.7-0.9 | Clear implication | Yes (default threshold) |
| 0.5-0.7 | Reasonable inference | Only if threshold lowered |
| < 0.5 | Uncertain | No, use defaults |

## Comparing Models

### RecovUS vs. Utility-Based

| Aspect | Utility-Based | RecovUS |
|--------|---------------|---------|
| Decision basis | Weighted utility function | Financial + community adequacy |
| Agent types | Uniform | 3 perception types |
| State tracking | Recovery level (0-1) | State machine |
| Financial modeling | None | 5-resource model |
| Transitions | Deterministic | Probabilistic |

### When to Use Each

**Use RecovUS when:**

- Studying equity in disaster recovery (income effects)
- Modeling realistic financial constraints
- Researching community-level recovery dynamics
- Comparing different perception type distributions

**Use Utility-Based when:**

- Quick prototyping or exploration
- Simpler scenarios without financial complexity
- Backward compatibility with earlier configurations

## Monitoring RecovUS Simulations

### State Distribution Logging

Every 5 steps, the simulation logs the state distribution:

```
INFO: Step 5: avg_recovery = 0.234, states = {'waiting': 35, 'repairing': 12, 'recovered': 3}
INFO: Step 10: avg_recovery = 0.456, states = {'waiting': 20, 'repairing': 18, 'recovered': 12}
INFO: Step 15: avg_recovery = 0.678, states = {'waiting': 8, 'repairing': 15, 'recovered': 27}
INFO: Step 20: avg_recovery = 0.823, states = {'waiting': 3, 'repairing': 8, 'recovered': 39}
```

### Accessing Household States

```python
result = engine.run()

# Check final states
for hh_id, household in engine._network.households.items():
    print(f"Household {hh_id}: "
          f"state={household.recovery_state}, "
          f"perception={household.perception_type}, "
          f"feasible={household.is_feasible}")
```

### Tracking Decision History

```python
# Each household tracks its decisions
household = engine._network.households[0]
print(f"Decision history: {household.decision_history}")
# ['waiting', 'waiting', 'started_repair', 'repairing', 'completed']
```

## Configuration via YAML

```yaml
# config.yaml
recovus:
  enabled: true

  # Perception distribution (must sum to 1.0)
  perception_infrastructure: 0.65
  perception_social: 0.31
  perception_community: 0.04

  # Adequacy thresholds
  adequacy_infrastructure: 0.50
  adequacy_neighbor: 0.40
  adequacy_community_assets: 0.50

  # Transition probabilities
  transition_r0: 0.35
  transition_r1: 0.95
  transition_r2: 0.95
  transition_relocate: 0.05

  # Financial parameters
  insurance_penetration_rate: 0.60
  fema_ha_max: 35500.0
  sba_loan_max: 200000.0
  sba_income_floor: 30000.0
  sba_uptake_rate: 0.40
```

## Disaster-Specific Recommendations

### Flood (e.g., Hurricane Harvey)

```python
flood_config = RecovUSConfig(
    perception_infrastructure=0.70,  # Higher infrastructure focus
    adequacy_infrastructure=0.55,    # Higher threshold needed
    insurance_penetration_rate=0.75, # NFIP coverage common
    transition_relocate=0.10,        # Some permanent displacement
)
```

### Hurricane (Wind Damage)

```python
hurricane_config = RecovUSConfig(
    perception_infrastructure=0.55,
    perception_social=0.40,          # Higher social influence
    adequacy_infrastructure=0.45,    # Faster infrastructure recovery
    transition_r1=0.98,              # Very high repair rate
)
```

### Earthquake

```python
earthquake_config = RecovUSConfig(
    perception_infrastructure=0.80,  # Structural concerns dominate
    adequacy_infrastructure=0.65,    # Higher safety threshold
    insurance_penetration_rate=0.20, # Low coverage
    transition_relocate=0.15,        # Higher relocation
)
```

### Wildfire

```python
wildfire_config = RecovUSConfig(
    perception_community=0.15,       # Higher community focus
    adequacy_community_assets=0.60,  # Businesses critical
    insurance_penetration_rate=0.40, # Variable coverage
    transition_relocate=0.20,        # High relocation rate
)
```

## References

- Moradi, S. & Nejat, A. (2020). RecovUS: An Agent-Based Model of Post-Disaster Household Recovery. *Journal of Artificial Societies and Social Simulation*, 23(4), 13. https://www.jasss.org/23/4/13.html
- FEMA Housing Assistance Program: https://www.fema.gov/assistance/individual/housing
- SBA Disaster Loans: https://www.sba.gov/funding-programs/disaster-assistance
