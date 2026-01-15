# Monte Carlo Analysis Guide

Learn how to run multiple simulations for statistical analysis.

## Why Monte Carlo?

Agent-based models have stochastic (random) elements:
- Random initial conditions (income, resilience)
- Probabilistic network connections
- Random seeds affecting behavior

A single run isn't statistically meaningful. Monte Carlo analysis runs many simulations to:
- Calculate mean outcomes and confidence intervals
- Understand variability in results
- Perform sensitivity analysis

## Basic Monte Carlo Run

### Command Line

```bash
# Run 100 simulations
python -m household_recovery --monte-carlo 100

# With parallel processing
python -m household_recovery --monte-carlo 100 --parallel
```

### Python API

```python
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.config import SimulationConfig

config = SimulationConfig(num_households=50, steps=20)

results = run_monte_carlo(
    config=config,
    n_runs=100,
    parallel=True
)

summary = results.get_summary()
print(f"Mean final recovery: {summary['final_recovery']['mean']:.3f}")
print(f"Std deviation: {summary['final_recovery']['std']:.3f}")
```

## Understanding Results

### Summary Statistics

```python
summary = results.get_summary()

# Returns:
{
    'n_runs': 100,
    'steps': 20,
    'final_recovery': {
        'mean': 0.723,      # Average final recovery
        'std': 0.045,       # Standard deviation
        'min': 0.612,       # Worst case
        'max': 0.831,       # Best case
        'median': 0.725,    # Middle value
        'ci_95': (0.638, 0.812)  # 95% confidence interval
    },
    'convergence': {
        'all_above_90': 0.15,   # 15% exceeded 90% recovery
        'all_above_80': 0.42,   # 42% exceeded 80%
        'all_above_50': 0.98    # 98% exceeded 50%
    }
}
```

### Confidence Intervals

The 95% CI tells you: "We're 95% confident the true mean falls in this range."

```python
ci_low, ci_high = summary['final_recovery']['ci_95']
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

### Trajectories

```python
import numpy as np

# Mean trajectory over time
mean_trajectory = results.mean_trajectory
print(f"Step 0: {mean_trajectory[0]:.3f}")
print(f"Step 10: {mean_trajectory[10]:.3f}")
print(f"Step 20: {mean_trajectory[20]:.3f}")

# Confidence bands
lower = results.ci_lower
upper = results.ci_upper
```

## Progress Tracking

### With tqdm

```python
from tqdm import tqdm

pbar = tqdm(total=100, desc="Monte Carlo")

def on_progress(completed, total):
    pbar.update(1)

results = run_monte_carlo(
    config=config,
    n_runs=100,
    progress_callback=on_progress
)

pbar.close()
```

### Simple Progress

```python
def on_progress(completed, total):
    if completed % 10 == 0:
        print(f"Completed {completed}/{total} runs")

results = run_monte_carlo(
    config=config,
    n_runs=100,
    progress_callback=on_progress
)
```

## Parallel Processing

Use multiple CPU cores:

```python
results = run_monte_carlo(
    config=config,
    n_runs=100,
    parallel=True,           # Enable parallelism
    max_workers=4            # Optional: limit workers
)
```

**Note**: Parallel runs use process-based parallelism, so each run is independent.

## Exporting Results

### Summary CSV

```python
results.export_summary_csv("mc_summary.csv")
```

Creates:
```csv
step,mean,std,ci_lower,ci_upper
0,0.0000,0.0000,0.0000,0.0000
1,0.0850,0.0120,0.0732,0.0968
...
```

### All Runs CSV

```python
results.export_all_runs_csv("mc_all_runs.csv")
```

Creates:
```csv
step,run_0,run_1,run_2,...
0,0.0000,0.0000,0.0000,...
1,0.0823,0.0891,0.0812,...
...
```

## Visualization

```python
from household_recovery.visualization import (
    plot_monte_carlo_trajectory,
    plot_recovery_distribution,
    create_monte_carlo_report
)

# Trajectory with confidence bands
plot_monte_carlo_trajectory(
    results,
    save_path="mc_trajectory.png",
    show_individual=True  # Show individual runs
)

# Distribution of final recovery
plot_recovery_distribution(
    results,
    save_path="mc_distribution.png"
)

# Complete report
create_monte_carlo_report(results, "./output/mc_analysis")
```

## Sensitivity Analysis

Test how different parameter values affect outcomes:

```python
from household_recovery.monte_carlo import sensitivity_analysis

base_config = SimulationConfig(num_households=50, steps=20)

# Test different recovery rates
results = sensitivity_analysis(
    base_config=base_config,
    parameter='base_recovery_rate',
    values=[0.05, 0.08, 0.10, 0.12, 0.15],
    n_runs_per_value=50,
    parallel=True
)

# Analyze results
for rate, mc_results in results.items():
    summary = mc_results.get_summary()
    mean = summary['final_recovery']['mean']
    std = summary['final_recovery']['std']
    print(f"Rate {rate:.2f}: {mean:.3f} ± {std:.3f}")
```

### Parameters to Test

Good candidates for sensitivity analysis:
- `base_recovery_rate`
- `num_households`
- `network_connectivity`
- `steps`

## Complete Example

```python
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.config import SimulationConfig, ThresholdConfig
from household_recovery.visualization import create_monte_carlo_report
from tqdm import tqdm

# Configure simulation
config = SimulationConfig(
    num_households=100,
    num_infrastructure=5,
    num_businesses=5,
    network_type='watts_strogatz',
    steps=30
)

thresholds = ThresholdConfig(
    income_low=40000,
    income_high=100000
)

# Run Monte Carlo with progress bar
pbar = tqdm(total=200, desc="Running simulations")

results = run_monte_carlo(
    config=config,
    n_runs=200,
    parallel=True,
    thresholds=thresholds,
    progress_callback=lambda i, n: pbar.update(1)
)

pbar.close()

# Analyze results
summary = results.get_summary()

print("\n=== Monte Carlo Results ===")
print(f"Runs: {summary['n_runs']}")
print(f"Final Recovery:")
print(f"  Mean: {summary['final_recovery']['mean']:.3f}")
print(f"  Std:  {summary['final_recovery']['std']:.3f}")
print(f"  95% CI: [{summary['final_recovery']['ci_95'][0]:.3f}, "
      f"{summary['final_recovery']['ci_95'][1]:.3f}]")
print(f"\nConvergence:")
print(f"  >90% recovery: {summary['convergence']['all_above_90']*100:.1f}%")
print(f"  >80% recovery: {summary['convergence']['all_above_80']*100:.1f}%")

# Generate report
create_monte_carlo_report(results, "./output/mc_analysis", prefix="study")
```

## How Many Runs?

| Runs | Precision | Use Case |
|------|-----------|----------|
| 10-30 | Low | Quick testing |
| 100 | Medium | Standard analysis |
| 500+ | High | Publication-quality |
| 1000+ | Very High | Detailed sensitivity |

More runs = narrower confidence intervals = more precise estimates.

## Tips

1. **Start small** - Test with 10 runs, then scale up
2. **Use parallel** - Much faster for large n_runs
3. **Set random seeds** - For base config reproducibility
4. **Export results** - Save data for later analysis
5. **Visualize** - Plots reveal patterns statistics miss

## Next Steps

- [Visualization](visualization.md) - Creating publication plots
- [Network Topologies](network-topologies.md) - Test different networks
- [Custom Parameters](custom-parameters.md) - Parameter tuning
