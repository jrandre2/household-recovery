# Decision Model Module

Abstract decision model protocol and implementations for household recovery decisions.

```python
from household_recovery.decision_model import (
    DecisionModel,
    UtilityDecisionModel,
    RecovUSDecisionModel,
    create_decision_model,
)
```

## Overview

This module provides a pluggable architecture for household decision-making:

- **DecisionModel**: Protocol defining the decision interface
- **UtilityDecisionModel**: Original utility-based model (backward compatible)
- **RecovUSDecisionModel**: RecovUS-style with feasibility, adequacy, and state machine

The decision model is selected via configuration, allowing easy comparison between different behavioral models.

---

## DecisionModel Protocol

Protocol interface for all decision models. Any custom decision model must implement this interface.

```python
@runtime_checkable
class DecisionModel(Protocol):
    def decide(
        self,
        household: HouseholdAgent,
        context: SimulationContext,
        heuristics: list[Heuristic],
        params: dict[str, Any],
    ) -> tuple[float, str]:
        ...
```

### decide()

Make a recovery decision for a household.

| Parameter | Type | Description |
|-----------|------|-------------|
| `household` | `HouseholdAgent` | The household agent making the decision |
| `context` | `SimulationContext` | Current simulation context (neighborhood state) |
| `heuristics` | `list[Heuristic]` | List of behavioral heuristics to apply |
| `params` | `dict[str, Any]` | Additional parameters (base_rate, weights, etc.) |

**Returns:** `tuple[float, str]` - New recovery level and action description

---

## UtilityDecisionModel

Original utility-based decision model for backward compatibility.

This model makes decisions based on a weighted utility function:

```
utility = w_self * recovery + w_neighbor * neighbors + w_infra * infra + w_business * business
```

Recovery is accepted if it improves utility. Heuristics modify the recovery increment via `boost` and `extra_recovery` multipliers.

### Constructor

```python
UtilityDecisionModel()
```

No parameters required.

### Methods

#### `decide(household, context, heuristics, params) -> tuple[float, str]`

Make utility-based recovery decision.

**Params dict keys:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_rate` | `float` | `0.1` | Base recovery increment per step |
| `weights` | `dict` | See below | Utility weights |

**Default weights:**
```python
{
    'self_recovery': 1.0,
    'neighbor_recovery': 0.3,
    'infrastructure': 0.2,
    'business': 0.2,
}
```

**Returns:**
- `('utility_increase', new_recovery)` if utility improves
- `('utility_no_change', current_recovery)` otherwise

### Heuristic Action Format

UtilityDecisionModel uses legacy heuristic actions:

| Action Key | Type | Description |
|------------|------|-------------|
| `boost` | `float` | Multiplier for base recovery rate |
| `extra_recovery` | `float` | Additive recovery bonus |

### Example

```python
from household_recovery.decision_model import UtilityDecisionModel

model = UtilityDecisionModel()
new_recovery, action = model.decide(
    household=agent,
    context=sim_context,
    heuristics=heuristics,
    params={'base_rate': 0.1}
)
```

---

## RecovUSDecisionModel

RecovUS-style decision model implementing the full Moradi & Nejat (2020) decision logic.

This model:
1. Checks financial feasibility (resources >= costs)
2. Checks community adequacy based on perception type
3. Uses state machine for probabilistic transitions
4. Applies heuristics to modify probabilities and thresholds

### Constructor

```python
@dataclass
class RecovUSDecisionModel:
    state_machine: RecoveryStateMachine
    base_probabilities: TransitionProbabilities
    base_criteria: CommunityAdequacyCriteria
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `state_machine` | `RecoveryStateMachine` | Manages state transitions |
| `base_probabilities` | `TransitionProbabilities` | Base transition probabilities (r0, r1, r2) |
| `base_criteria` | `CommunityAdequacyCriteria` | Adequacy thresholds by perception type |

### Methods

#### `decide(household, context, heuristics, params) -> tuple[float, str]`

Make RecovUS-style recovery decision.

**Decision Flow:**
1. Build extended context with RecovUS fields
2. Apply heuristics to modify probabilities and thresholds
3. Check financial feasibility (5 resources vs. 2 costs)
4. Check community adequacy based on perception type
5. Use state machine for probabilistic transition
6. Update household state and return new recovery

**Returns:** `tuple[float, str]` - New recovery level and action (e.g., `'start_repair'`, `'continue_repair'`, `'complete_recovery'`, `'relocate'`)

### Heuristic Action Format

RecovUSDecisionModel uses a new action format:

| Action Key | Type | Description |
|------------|------|-------------|
| `modify_r0` | `float` | Multiplier for r0 (waiting -> repairing) probability |
| `modify_r1` | `float` | Multiplier for r1 (continue repairing) probability |
| `modify_r2` | `float` | Multiplier for r2 (repairing -> recovered) probability |
| `modify_adq_infr` | `float` | Additive change to infrastructure adequacy threshold |
| `modify_adq_nbr` | `float` | Additive change to neighbor adequacy threshold |
| `modify_adq_cas` | `float` | Additive change to community assets adequacy threshold |

**Example heuristic:**
```python
Heuristic(
    condition_str="income < 30000 and perception_type == 'infrastructure'",
    action={
        'modify_r0': 0.8,  # 20% slower to start repairs
        'modify_adq_infr': 0.1,  # Higher infrastructure threshold
    },
    source="Low-income infrastructure perception study"
)
```

### Extended Context Fields

The RecovUSDecisionModel adds these fields to the context dict:

| Field | Type | Description |
|-------|------|-------------|
| `perception_type` | `str` | Household's ASNA perception type |
| `damage_severity` | `str` | Damage level (`'minor'`, `'moderate'`, `'severe'`) |
| `recovery_state` | `str` | Current state (`'waiting'`, `'repairing'`, `'recovered'`, `'relocated'`) |
| `repair_cost` | `float` | Estimated repair cost |
| `is_habitable` | `bool` | Whether home is currently habitable |
| `available_resources` | `float` | Sum of all financial resources |
| `is_feasible` | `bool` | Whether resources >= costs |

### Example

```python
from household_recovery.decision_model import create_decision_model

model = create_decision_model(
    model_type='recovus',
    rng=np.random.default_rng(42)
)

new_recovery, action = model.decide(
    household=agent,
    context=sim_context,
    heuristics=recovus_heuristics,
    params={'base_rate': 0.02}
)

print(f"Action: {action}, New recovery: {new_recovery:.3f}")
```

---

## create_decision_model()

Factory function to create a decision model.

```python
def create_decision_model(
    model_type: str = 'utility',
    rng: np.random.Generator | None = None,
    **kwargs,
) -> DecisionModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | `'utility'` | Model type: `'utility'` or `'recovus'` |
| `rng` | `Generator` | `None` | Random generator for RecovUS model |
| `probabilities` | `TransitionProbabilities` | `None` | Custom transition probabilities |
| `criteria` | `CommunityAdequacyCriteria` | `None` | Custom adequacy criteria |

### Example

```python
from household_recovery.decision_model import create_decision_model
import numpy as np

# Create utility model (legacy)
utility_model = create_decision_model(model_type='utility')

# Create RecovUS model with defaults
recovus_model = create_decision_model(
    model_type='recovus',
    rng=np.random.default_rng(42)
)

# Create RecovUS model with custom parameters
from household_recovery.recovus import TransitionProbabilities, CommunityAdequacyCriteria

recovus_model = create_decision_model(
    model_type='recovus',
    rng=np.random.default_rng(42),
    probabilities=TransitionProbabilities(r0=0.40, r1=0.90, r2=0.85),
    criteria=CommunityAdequacyCriteria(infrastructure=0.60, neighbor=0.50)
)
```

---

## See Also

- [RecovUS Module](recovus.md) - Full RecovUS implementation details
- [Agents Module](agents.md) - HouseholdAgent and SimulationContext
- [Heuristics Module](heuristics.md) - Heuristic extraction and application
- [Configuration](config.md) - RecovUSConfig for model selection
