# Agents Module

Agent classes for households, infrastructure, and businesses.

```python
from household_recovery.agents import (
    HouseholdAgent,
    InfrastructureNode,
    BusinessNode,
    SimulationContext
)
```

## Type Aliases

```python
IncomeLevel = Literal['low', 'middle', 'high']
ResilienceCategory = Literal['low', 'medium', 'high']
```

---

## HouseholdAgent

Represents a household in the disaster recovery simulation.

Households are the primary agents that make recovery decisions based on:
- Individual characteristics (income, resilience)
- State of their neighbors
- State of infrastructure and businesses
- Behavioral heuristics from research

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `int` | Unique identifier |
| `income` | `float` | Annual household income in dollars |
| `income_level` | `IncomeLevel` | Categorical: 'low', 'middle', 'high' |
| `resilience` | `float` | Resilience score (0-1) |
| `resilience_category` | `ResilienceCategory` | Categorical: 'low', 'medium', 'high' |
| `recovery` | `float` | Current recovery level (0=not recovered, 1=fully) |
| `recovery_history` | `list[float]` | Recovery values at each step |

### Class Methods

#### `generate_random(agent_id, rng=None, thresholds=None) -> HouseholdAgent`

Generate a household with random but realistic attributes.

```python
import numpy as np
from household_recovery.agents import HouseholdAgent
from household_recovery.config import ThresholdConfig

rng = np.random.default_rng(42)
thresholds = ThresholdConfig()

household = HouseholdAgent.generate_random(
    agent_id=0,
    rng=rng,
    thresholds=thresholds
)
```

**Distribution Details:**
- Income: Log-normal distribution (median ~$57k, mean ~$80k)
- Resilience: Beta distribution (0-1, moderate-low average)

### Instance Methods

#### `calculate_utility(proposed_recovery, context, weights=None) -> float`

Calculate utility for a proposed recovery level.

```python
utility = household.calculate_utility(
    proposed_recovery=0.5,
    context=simulation_context,
    weights={
        'self_recovery': 1.0,
        'neighbor_recovery': 0.3,
        'infrastructure': 0.2,
        'business': 0.2
    }
)
```

**Utility Formula:**
```
utility = w_self * own_recovery
        + w_neighbor * avg_neighbor_recovery
        + w_infra * infrastructure_functionality
        + w_business * business_availability
```

#### `decide_recovery(context, heuristics, base_rate=0.1, utility_weights=None) -> float`

Decide the new recovery level based on context and heuristics.

```python
new_recovery = household.decide_recovery(
    context=context,
    heuristics=heuristics,
    base_rate=0.1
)
```

**Process:**
1. Evaluate all heuristics against current context
2. Aggregate boosts and extra recovery from matched heuristics
3. Calculate proposed new recovery
4. Accept if utility increases

#### `record_state()`

Record current recovery level in history.

---

## InfrastructureNode

Represents infrastructure (power, water, roads, etc.).

Infrastructure functionality affects household recovery ability and improves as households recover.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., 'infra_0') |
| `functionality` | `float` | Functionality level (0-1) |
| `functionality_history` | `list[float]` | History of functionality values |

### Class Methods

#### `generate_random(node_id, rng=None, infra_config=None) -> InfrastructureNode`

Generate infrastructure with random initial functionality.

```python
infra = InfrastructureNode.generate_random(
    node_id="infra_0",
    rng=rng,
    infra_config=InfrastructureConfig()
)
```

### Instance Methods

#### `update(connected_households, improvement_rate=0.05, household_recovery_multiplier=0.1)`

Update functionality based on connected household recovery.

```python
infra.update(
    connected_households=[hh1, hh2, hh3],
    improvement_rate=0.05,
    household_recovery_multiplier=0.1
)
```

#### `record_state()`

Record current functionality in history.

---

## BusinessNode

Represents local businesses (shops, services, employers).

Business availability provides economic incentive for recovery.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., 'business_0') |
| `availability` | `float` | Availability level (0-1) |
| `availability_history` | `list[float]` | History of availability values |

### Class Methods

#### `generate_random(node_id, rng=None, infra_config=None) -> BusinessNode`

Generate business with random initial availability.

### Instance Methods

#### `update(connected_households, improvement_rate=0.05, household_recovery_multiplier=0.1)`

Update availability based on connected household recovery.

#### `record_state()`

Record current availability in history.

---

## SimulationContext

Context passed to agents and heuristics during simulation.

Contains neighborhood-level aggregate information for recovery decisions.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `avg_neighbor_recovery` | `float` | 0.0 | Average neighbor recovery |
| `avg_infra_func` | `float` | 0.0 | Average infrastructure functionality |
| `avg_business_avail` | `float` | 0.0 | Average business availability |
| `num_neighbors` | `int` | 0 | Number of neighbors |
| `resilience` | `float` | 0.5 | Household resilience |
| `resilience_category` | `str` | 'medium' | Resilience category |
| `household_income` | `float` | 60000.0 | Household income |
| `income_level` | `str` | 'middle' | Income level category |

### Methods

#### `to_dict() -> dict`

Convert to dictionary for heuristic evaluation.

```python
ctx = SimulationContext(
    avg_neighbor_recovery=0.6,
    avg_infra_func=0.4,
    resilience=0.7,
    income_level='middle'
)

ctx_dict = ctx.to_dict()
# Used by heuristics: ctx['avg_neighbor_recovery'] > 0.5
```
