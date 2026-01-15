# Agent Model Design

Design decisions for the agent-based modeling components.

## Agent Types

### HouseholdAgent

Primary decision-making agents in the simulation.

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | int | Unique identifier |
| `income` | float | Annual income (log-normal distribution) |
| `income_level` | str | 'low', 'middle', 'high' |
| `resilience` | float | Resilience score 0-1 (beta distribution) |
| `resilience_category` | str | 'low', 'medium', 'high' |
| `recovery` | float | Current recovery 0-1 |

**Generation:**
```python
# Income: log-normal (median ~$57k, mean ~$80k)
income = rng.lognormal(10.95, 0.82)

# Resilience: beta distribution (moderate-low average)
resilience = rng.beta(3.2, 4.8)
```

### InfrastructureNode

Represents critical infrastructure (power, water, roads).

**Dynamics:**
- Improves over time (base improvement rate)
- Improves faster when connected households recover
- Affects household recovery capability

### BusinessNode

Represents local economic activity.

**Dynamics:**
- Similar to infrastructure
- Provides economic incentive for recovery

## Decision Process

### 1. Context Building

For each household, the network builds a `SimulationContext`:

```python
context = SimulationContext(
    avg_neighbor_recovery=mean([h.recovery for h in neighbors]),
    avg_infra_func=mean([i.functionality for i in connected_infra]),
    avg_business_avail=mean([b.availability for b in connected_biz]),
    num_neighbors=len(neighbors),
    resilience=household.resilience,
    resilience_category=household.resilience_category,
    household_income=household.income,
    income_level=household.income_level
)
```

### 2. Heuristic Evaluation

Each heuristic is evaluated against the context:

```python
boost = 1.0
extra_recovery = 0.0

for heuristic in heuristics:
    ctx_dict = context.to_dict()
    if heuristic.evaluate(ctx_dict):
        boost *= heuristic.action.get('boost', 1.0)
        extra_recovery += heuristic.action.get('extra_recovery', 0.0)
```

### 3. Recovery Calculation

```python
increment = base_recovery_rate * boost + extra_recovery
proposed_recovery = min(current_recovery + increment, 1.0)
```

### 4. Utility Evaluation

```python
def calculate_utility(proposed_recovery, context, weights):
    return (
        weights['self_recovery'] * proposed_recovery +
        weights['neighbor_recovery'] * context.avg_neighbor_recovery +
        weights['infrastructure'] * context.avg_infra_func +
        weights['business'] * context.avg_business_avail
    )

current_utility = calculate_utility(current_recovery, context, weights)
proposed_utility = calculate_utility(proposed_recovery, context, weights)
```

### 5. Decision

```python
if proposed_utility > current_utility:
    household.recovery = proposed_recovery
# else: no change
```

## Utility Function Design

### Rationale

Households don't blindly increase recovery. They evaluate:
- **Self-interest**: Higher recovery is better for themselves
- **Social pressure**: Neighbors recovering creates incentive
- **Access to services**: Infrastructure and business availability matters

### Default Weights

```python
utility_weights = {
    'self_recovery': 1.0,      # Primary driver
    'neighbor_recovery': 0.3,  # Social influence
    'infrastructure': 0.2,     # Service access
    'business': 0.2            # Economic opportunity
}
```

### Interpretation

- High `self_recovery`: Individual-focused recovery
- High `neighbor_recovery`: Community-driven recovery
- High `infrastructure`: Infrastructure-dependent recovery

## Heuristic Actions

### Boost

Multiplies the base recovery rate:

```python
# Positive boost (accelerate)
{'boost': 1.5}  # 50% faster recovery

# Negative boost (decelerate)
{'boost': 0.6}  # 40% slower recovery
```

### Extra Recovery

Adds directly to recovery increment:

```python
{'extra_recovery': 0.1}  # Additional 0.1 recovery per step
```

### Combined

```python
increment = base_rate * boost + extra_recovery
# With base_rate=0.1, boost=1.5, extra=0.05:
# increment = 0.1 * 1.5 + 0.05 = 0.2
```

## Network Effects

### Household-Household

```
    H1 ── H2
    │     │
    H3 ── H4
```

- Neighbors influence each other through `avg_neighbor_recovery`
- Network topology determines neighbor sets

### Household-Infrastructure

```
    I1
   /│\
  H1 H2 H3
```

- Infrastructure functionality affects all connected households
- Connection is probabilistic based on `connection_probability`

### Feedback Loops

1. **Positive**: Recovering households → Better infrastructure → Faster recovery
2. **Negative**: Poor infrastructure → Slow recovery → Infrastructure stays poor

## Classification Thresholds

### Income Classification

```python
if income < income_low:      # e.g., < $45,000
    level = 'low'
elif income < income_high:   # e.g., < $120,000
    level = 'middle'
else:
    level = 'high'
```

### Resilience Classification

```python
if resilience < resilience_low:    # e.g., < 0.35
    category = 'low'
elif resilience < resilience_high:  # e.g., < 0.70
    category = 'medium'
else:
    category = 'high'
```

### Why Classify?

Heuristics can target specific groups:

```python
# Target vulnerable populations
"ctx['income_level'] == 'low' and ctx['resilience_category'] == 'low'"

# Target high-capacity households
"ctx['income_level'] == 'high' and ctx['resilience'] > 0.7"
```

## Emergent Behavior

### Recovery Clustering

Neighbors tend to recover together due to:
- Shared infrastructure access
- Social influence via utility weights
- Similar heuristic conditions

### Inequality

Without intervention, high-income/high-resilience households recover faster:
- Better initial conditions
- More likely to meet positive heuristic conditions
- Network position effects (scale-free networks favor hubs)

### Tipping Points

Once average recovery exceeds certain thresholds:
- More heuristics become active
- Positive feedback accelerates recovery
- The community can "take off"

## Design Principles

### 1. Bounded Rationality

Agents don't optimize globally - they evaluate local information:
- Only neighbor recovery, not community-wide
- Only connected infrastructure, not all infrastructure

### 2. Heterogeneity

No two households are identical:
- Random attribute generation
- Different network positions
- Different heuristic evaluations

### 3. Emergence

Community-level patterns emerge from individual decisions:
- Recovery trajectories
- Inequality patterns
- Spatial clustering

### 4. Research Grounding

Behavior driven by academic research:
- RAG-extracted heuristics
- Empirically-based parameters
- Literature-informed thresholds

## Next Steps

- [Security](security.md) - Safe heuristic evaluation
- [RAG Architecture](rag-architecture.md) - Heuristic extraction
- [Data Flow](data-flow.md) - Simulation flow
