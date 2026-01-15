# Custom Parameters Guide

Learn how to customize simulation parameters for your specific use case.

## Configuration Methods

There are three ways to configure simulations:

1. **Command line arguments** - Quick testing
2. **Configuration files** - Reproducible experiments
3. **Python API** - Programmatic control

## Using Configuration Files

### Creating a Config File

Create `config.yaml`:

```yaml
simulation:
  num_households: 100
  num_infrastructure: 5
  num_businesses: 5
  network_type: watts_strogatz
  steps: 30
  base_recovery_rate: 0.08

thresholds:
  income:
    low: 35000
    high: 90000
  resilience:
    low: 0.25
    high: 0.65
```

### Running with Config

```bash
python -m household_recovery --config config.yaml
```

Or in Python:

```python
from household_recovery.config import FullConfig
from household_recovery import SimulationEngine

config = FullConfig.from_file("config.yaml")
engine = SimulationEngine(
    config=config.simulation,
    thresholds=config.thresholds
)
result = engine.run()
```

## Key Parameters to Tune

### Simulation Size

| Parameter | Effect |
|-----------|--------|
| `num_households` | More agents = more realistic but slower |
| `num_infrastructure` | More services = more connections |
| `num_businesses` | More economic opportunities |
| `steps` | Longer simulations = higher final recovery |

**Example:**

```yaml
simulation:
  num_households: 200   # Large community
  num_infrastructure: 8  # Multiple utilities
  num_businesses: 10     # Diverse economy
  steps: 50              # Long-term recovery
```

### Recovery Rate

`base_recovery_rate` controls how fast households recover per step.

| Value | Speed | Best For |
|-------|-------|----------|
| 0.05 | Slow | Long-term studies |
| 0.1 | Medium | Default behavior |
| 0.15 | Fast | Short simulations |

### Utility Weights

Control what factors influence recovery decisions:

```yaml
simulation:
  utility_weights:
    self_recovery: 1.0       # Own recovery importance
    neighbor_recovery: 0.5   # Social influence (high)
    infrastructure: 0.3      # Infrastructure dependence
    business: 0.2            # Economic factors
```

**High neighbor_recovery** = Strong community effects
**High infrastructure** = Critical infrastructure dependency

### Classification Thresholds

Define what "low" and "high" mean for income and resilience:

```yaml
thresholds:
  income:
    low: 30000     # Poverty line
    high: 100000   # Upper middle class
  resilience:
    low: 0.3       # Vulnerable population
    high: 0.7      # Highly resilient
```

These thresholds affect how heuristics classify households.

## Network Configuration

### Network Type

Choose topology based on your research question:

```yaml
simulation:
  network_type: watts_strogatz  # Small-world
  network_connectivity: 4       # Avg connections
```

| Type | Description | When to Use |
|------|-------------|-------------|
| `barabasi_albert` | Scale-free, hubs | Social networks |
| `watts_strogatz` | Small-world | Information spread |
| `erdos_renyi` | Random | Baseline comparison |
| `random_geometric` | Spatial | Geographic studies |

### Connection Probability

How likely households are to connect to infrastructure/businesses:

```yaml
network:
  connection_probability: 0.7  # 70% connection chance
```

Higher values = more connected community

## Infrastructure Parameters

```yaml
infrastructure:
  improvement_rate: 0.06          # How fast infra recovers
  initial_functionality_min: 0.1  # Severe damage
  initial_functionality_max: 0.4  # Moderate damage
  household_recovery_multiplier: 0.15  # Feedback strength
```

## Parameter Precedence

When using RAG-extracted parameters, precedence is:

1. **RAG-extracted** (if confidence >= 0.7)
2. **Config file** values
3. **Hardcoded defaults**

Override RAG values by setting explicit config values.

## Common Configurations

### Quick Test

```yaml
simulation:
  num_households: 20
  steps: 10
```

### Publication-Ready

```yaml
simulation:
  num_households: 100
  num_infrastructure: 5
  num_businesses: 5
  network_type: watts_strogatz
  network_connectivity: 4
  steps: 30
  random_seed: 42  # Reproducible
  base_recovery_rate: 0.08

infrastructure:
  improvement_rate: 0.05
  initial_functionality_min: 0.2
  initial_functionality_max: 0.5
```

### Vulnerability Study

Focus on low-income, low-resilience populations:

```yaml
thresholds:
  income:
    low: 25000    # Lower poverty line
    high: 60000   # Lower middle class ceiling
  resilience:
    low: 0.4      # More people classified as low
    high: 0.8     # Fewer classified as high

simulation:
  utility_weights:
    neighbor_recovery: 0.5   # Strong community effects
    infrastructure: 0.4      # High infrastructure dependency
```

### Network Effects Study

Emphasize social connections:

```yaml
simulation:
  network_type: watts_strogatz
  network_connectivity: 6  # Higher connectivity
  utility_weights:
    neighbor_recovery: 0.6  # Strong social influence

network:
  connection_probability: 0.8  # Dense connections
```

## Validating Configuration

```python
from household_recovery.config import FullConfig

config = FullConfig.from_file("config.yaml")

# Validate all sections
try:
    config.validate()
    print("Configuration is valid")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Programmatic Configuration

```python
from household_recovery.config import (
    SimulationConfig,
    ThresholdConfig,
    InfrastructureConfig,
    NetworkConfig
)

# Build configuration programmatically
sim_config = SimulationConfig(
    num_households=100,
    steps=30,
    network_type='watts_strogatz',
    base_recovery_rate=0.08,
    utility_weights={
        'self_recovery': 1.0,
        'neighbor_recovery': 0.4,
        'infrastructure': 0.3,
        'business': 0.2
    }
)

thresholds = ThresholdConfig(
    income_low=35000,
    income_high=90000,
    resilience_low=0.3,
    resilience_high=0.7
)

# Use in simulation
engine = SimulationEngine(
    config=sim_config,
    thresholds=thresholds
)
```

## Next Steps

- [Network Topologies](network-topologies.md) - Understand network options
- [RAG Pipeline](rag-pipeline.md) - Research-based parameters
- [Monte Carlo](monte-carlo.md) - Test parameter sensitivity
