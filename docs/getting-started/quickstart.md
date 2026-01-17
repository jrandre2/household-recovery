# Quickstart Guide

Get a simulation running in 5 minutes.

## Command Line Usage

### Basic Simulation

Run with default settings:

```bash
python -m household_recovery
```

### Custom Parameters

Specify number of households and simulation steps:

```bash
python -m household_recovery --households 50 --steps 20
```

### With Visualization

Save plots to an output directory:

```bash
python -m household_recovery --households 30 --steps 15 --output ./results
```

### Monte Carlo Analysis

Run multiple simulations for statistical analysis:

```bash
python -m household_recovery --monte-carlo 100 --parallel
```

## Python API Usage

### Minimal Example

```python
from household_recovery import SimulationEngine, SimulationConfig

# Create configuration
config = SimulationConfig(
    num_households=30,
    steps=15
)

# Run simulation
engine = SimulationEngine(config)
result = engine.run()

# View results
print(f"Final recovery: {result.final_recovery:.3f}")
print(f"Duration: {result.duration_seconds:.2f}s")
```

### With Custom Thresholds

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import ThresholdConfig

config = SimulationConfig(
    num_households=50,
    steps=20,
    network_type='watts_strogatz'
)

thresholds = ThresholdConfig(
    income_low=40000,      # Below this = 'low' income
    income_high=100000,    # Above this = 'high' income
    resilience_low=0.3,    # Below this = 'low' resilience
    resilience_high=0.7    # Above this = 'high' resilience
)

engine = SimulationEngine(config, thresholds=thresholds)
result = engine.run()
```

### Using a Config File

```python
from household_recovery import SimulationEngine
from household_recovery.config import FullConfig

# Load from YAML
full_config = FullConfig.from_file("config.yaml")

engine = SimulationEngine(
    config=full_config.simulation,
    thresholds=full_config.thresholds,
    infra_config=full_config.infrastructure,
    network_config=full_config.network
)

result = engine.run()
```

## Understanding the Output

### Recovery Trajectory

The simulation tracks average recovery level at each step:

```
Step 0: avg_recovery = 0.000   # Initial state (all households at 0)
Step 1: avg_recovery = 0.085   # Early recovery
Step 2: avg_recovery = 0.156
...
Step 15: avg_recovery = 0.723  # Final recovery level
```

### Result Object

The `SimulationResult` contains:

| Property | Description |
|----------|-------------|
| `final_recovery` | Final average recovery (0-1) |
| `recovery_history` | List of recovery levels per step |
| `num_steps` | Number of steps completed |
| `duration_seconds` | Wall-clock time |
| `heuristics_used` | Behavioral rules applied |

### Exporting Results

```python
# Export to CSV (for analysis)
result.export_csv("results.csv")

# Export to JSON (for reproducibility)
result.export_json("results.json")
```

## Network Types

Try different network topologies:

```bash
# Scale-free network (realistic social networks)
python -m household_recovery --network barabasi_albert

# Small-world network (high clustering)
python -m household_recovery --network watts_strogatz

# Random network (baseline)
python -m household_recovery --network erdos_renyi

# Spatial network (geographic)
python -m household_recovery --network random_geometric
```

## Using RAG Pipeline

If you have API keys configured:

```bash
# Use Google Scholar papers
python -m household_recovery --serpapi-key YOUR_KEY --groq-key YOUR_KEY

# Use local PDFs
python -m household_recovery --pdf-dir ~/research/papers --groq-key YOUR_KEY
```

## Next Steps

- [Configuration Reference](configuration.md) - Full parameter documentation
- [Basic Simulation Guide](../user-guide/basic-simulation.md) - Detailed walkthrough
- [API Reference](../api-reference/index.md) - Complete API documentation
