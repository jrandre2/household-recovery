# Basic Simulation Tutorial

Learn how to run your first household recovery simulation.

## What You'll Learn

- How the simulation works
- Running simulations from command line and Python
- Understanding simulation output
- Interpreting recovery trajectories

## How the Simulation Works

The simulation models disaster recovery as an agent-based model:

1. **Households** are created with random income and resilience values
2. **Infrastructure** (power, water) and **Businesses** are added
3. **Network connections** link households to each other and to services
4. **Each step**, households decide whether to increase their recovery based on:
   - Their individual characteristics
   - Neighbor recovery status
   - Infrastructure functionality
   - Business availability
   - Behavioral heuristics from research

**Note:** RecovUS is enabled by default. The utility-based decision details below apply when RecovUS is disabled.

## Running Your First Simulation

### Command Line

```bash
# Basic run with defaults
python -m household_recovery

# Custom parameters
python -m household_recovery --households 30 --steps 15
```

### Python API

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
```

## Understanding the Output

### Step-by-Step Progress

```
Step 0: avg_recovery = 0.000   # Initial state
Step 1: avg_recovery = 0.085   # Early recovery begins
Step 2: avg_recovery = 0.156
Step 3: avg_recovery = 0.225
...
Step 15: avg_recovery = 0.723  # Final state
```

- **Step 0**: All households start at 0% recovery
- **Each step**: Households evaluate their situation and potentially increase recovery
- **Final**: Average recovery across all households

### Result Object

```python
# Key properties
result.final_recovery      # 0.723 - final average (0-1)
result.recovery_history    # [0.0, 0.085, 0.156, ...] - trajectory
result.num_steps           # 15 - steps completed
result.duration_seconds    # 0.45 - wall-clock time

# Access individual household data
trajectories = result.get_household_trajectories()
for hh_id, history in trajectories.items():
    print(f"Household {hh_id}: started {history[0]:.2f}, ended {history[-1]:.2f}")
```

### Statistics

```python
stats = result.get_final_statistics()

print(f"Recovery - Mean: {stats['recovery']['mean']:.3f}")
print(f"Recovery - Std:  {stats['recovery']['std']:.3f}")
print(f"Recovery - Min:  {stats['recovery']['min']:.3f}")
print(f"Recovery - Max:  {stats['recovery']['max']:.3f}")
```

## Exporting Results

### CSV Export

```python
result.export_csv("results.csv")
```

Creates a file with columns:
```
step,avg_recovery,household_0,household_1,...
0,0.000,0.00,0.00,...
1,0.085,0.10,0.08,...
```

### JSON Export

```python
result.export_json("results.json")
```

Creates detailed metadata including:
- Configuration used
- Start/end times
- Random seed
- Heuristics applied
- Final statistics

## Understanding Recovery Dynamics

### Why Recovery Varies

Households recover at different rates due to:

1. **Income level**: Higher income households may recover faster
2. **Resilience**: More resilient households bounce back quicker
3. **Network position**: Well-connected households benefit from neighbor recovery
4. **Infrastructure access**: Households near functional infrastructure recover faster

### Utility-Based Decisions

Households only increase recovery if it improves their "utility":

```
utility = 1.0 * own_recovery
        + 0.3 * avg_neighbor_recovery
        + 0.2 * infrastructure_functionality
        + 0.2 * business_availability
```

Recovery only happens when the proposed new recovery level would increase utility.

### Heuristic Boosts

Research-based heuristics modify recovery rates:

```
IF avg_neighbor_recovery > 0.5 THEN boost recovery by 50%
IF avg_infra_func < 0.3 THEN reduce recovery by 40%
```

## Example: Complete Workflow

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.visualization import (
    plot_recovery_trajectory,
    create_simulation_report
)

# Configure simulation
config = SimulationConfig(
    num_households=50,
    num_infrastructure=3,
    num_businesses=3,
    steps=20,
    random_seed=42  # For reproducibility
)

# Run with progress tracking
def on_progress(step, recovery):
    print(f"Step {step}: {recovery:.3f}")

engine = SimulationEngine(config)
result = engine.run(progress_callback=on_progress)

# Analyze results
print(f"\n=== Results ===")
print(f"Final recovery: {result.final_recovery:.3f}")
print(f"Duration: {result.duration_seconds:.2f}s")

stats = result.get_final_statistics()
print(f"Min household: {stats['recovery']['min']:.3f}")
print(f"Max household: {stats['recovery']['max']:.3f}")

# Export and visualize
create_simulation_report(result, "./output", prefix="my_simulation")

print("\nCreated files in ./output/:")
print("  - my_simulation_trajectory.png")
print("  - my_simulation_network_final.png")
print("  - my_simulation_data.csv")
print("  - my_simulation_metadata.json")
```

## Troubleshooting

### Low Recovery

If final recovery is very low:
- Increase `steps` to give more time
- Increase `base_recovery_rate`
- Check that infrastructure has reasonable initial functionality

### High Variability

If different runs give very different results:
- Use `random_seed` for reproducibility
- Consider running Monte Carlo analysis for statistics

### Slow Performance

For large simulations:
- Reduce `num_households` for testing
- Use `parallel=True` with Monte Carlo

## Next Steps

- [Custom Parameters](custom-parameters.md) - Customize your simulation
- [RAG Pipeline](rag-pipeline.md) - Use research-based heuristics
- [Monte Carlo](monte-carlo.md) - Statistical analysis
