# Monte Carlo Module

Multi-run statistical analysis.

```python
from household_recovery.monte_carlo import (
    MonteCarloResults,
    run_monte_carlo,
    sensitivity_analysis
)
```

## Why Monte Carlo?

Monte Carlo methods use repeated random sampling to obtain numerical results. In agent-based modeling, this is essential because:

1. **Stochastic elements** (random initial conditions, probabilistic decisions) mean single runs are not representative
2. **Running N simulations** lets us calculate mean trajectories, confidence intervals, and identify outliers
3. **Statistical significance** requires multiple runs to distinguish real effects from random variation

---

## MonteCarloResults

Aggregated results from multiple simulation runs.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `individual_results` | `list[SimulationResult]` | All individual run results |
| `config` | `SimulationConfig` | Configuration used |
| `n_runs` | `int` | Number of runs |

### Properties

#### `mean_trajectory -> np.ndarray`

Mean recovery trajectory across all runs.

#### `std_trajectory -> np.ndarray`

Standard deviation of recovery trajectory.

#### `ci_lower -> np.ndarray`

Lower bound of 95% confidence interval.

#### `ci_upper -> np.ndarray`

Upper bound of 95% confidence interval.

#### `final_recovery_distribution -> np.ndarray`

Array of final recovery values from all runs.

### Methods

#### `get_summary() -> dict`

Get summary statistics.

```python
summary = results.get_summary()

# Returns:
{
    'n_runs': 100,
    'steps': 20,
    'final_recovery': {
        'mean': 0.723,
        'std': 0.045,
        'min': 0.612,
        'max': 0.831,
        'median': 0.725,
        'ci_95': (0.638, 0.812)
    },
    'convergence': {
        'all_above_90': 0.15,   # 15% of runs exceeded 90% recovery
        'all_above_80': 0.42,   # 42% exceeded 80%
        'all_above_50': 0.98    # 98% exceeded 50%
    }
}
```

#### `export_summary_csv(filepath)`

Export mean trajectory with confidence intervals.

```python
results.export_summary_csv("mc_summary.csv")
```

**Output columns:** `step, mean, std, ci_lower, ci_upper`

#### `export_all_runs_csv(filepath)`

Export all individual run trajectories.

```python
results.export_all_runs_csv("mc_all_runs.csv")
```

**Output columns:** `step, run_0, run_1, ..., run_n`

---

## run_monte_carlo

Run multiple simulations and aggregate results.

```python
results = run_monte_carlo(
    config=SimulationConfig(num_households=50, steps=20),
    n_runs=100,
    api_config=None,
    research_config=None,
    heuristics=None,
    parallel=True,
    max_workers=None,
    progress_callback=None,
    thresholds=None,
    infra_config=None,
    network_config=None,
    recovus_config=None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `SimulationConfig` | required | Base simulation config |
| `n_runs` | `int` | 100 | Number of simulation runs |
| `api_config` | `APIConfig` | `None` | API config for RAG |
| `research_config` | `ResearchConfig` | `None` | Paper retrieval config |
| `heuristics` | `list[Heuristic]` | `None` | Pre-built heuristics |
| `parallel` | `bool` | `False` | Use parallel execution |
| `max_workers` | `int` | `None` | Max parallel workers |
| `progress_callback` | `callable` | `None` | Called with `(run_num, n_runs)` |
| `thresholds` | `ThresholdConfig` | `None` | Classification thresholds |
| `infra_config` | `InfrastructureConfig` | `None` | Infrastructure params |
| `network_config` | `NetworkConfig` | `None` | Network params |
| `recovus_config` | `RecovUSConfig` | `None` | RecovUS decision model config |

### Example

```python
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.config import SimulationConfig
from tqdm import tqdm

config = SimulationConfig(num_households=50, steps=20)

# With progress bar
pbar = tqdm(total=100)
def on_progress(i, n):
    pbar.update(1)

results = run_monte_carlo(
    config=config,
    n_runs=100,
    parallel=True,
    progress_callback=on_progress
)

pbar.close()

summary = results.get_summary()
print(f"Final recovery: {summary['final_recovery']['mean']:.3f} ± {summary['final_recovery']['std']:.3f}")
```

---

## sensitivity_analysis

Run sensitivity analysis on a single parameter.

```python
results = sensitivity_analysis(
    base_config=SimulationConfig(),
    parameter='base_recovery_rate',
    values=[0.05, 0.1, 0.15, 0.2],
    n_runs_per_value=10,
    parallel=True
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_config` | `SimulationConfig` | Base configuration |
| `parameter` | `str` | Parameter name to vary |
| `values` | `list[Any]` | Values to test |
| `n_runs_per_value` | `int` | Monte Carlo runs per value |
| `**kwargs` | | Additional args for `run_monte_carlo` |

### Returns

`dict[Any, MonteCarloResults]` - Maps parameter values to results.

### Example

```python
from household_recovery.monte_carlo import sensitivity_analysis
from household_recovery.config import SimulationConfig

base = SimulationConfig(num_households=50, steps=20)

# Test different recovery rates
results = sensitivity_analysis(
    base_config=base,
    parameter='base_recovery_rate',
    values=[0.05, 0.08, 0.1, 0.12, 0.15],
    n_runs_per_value=20,
    parallel=True
)

# Analyze results
for rate, mc_results in results.items():
    summary = mc_results.get_summary()
    mean = summary['final_recovery']['mean']
    std = summary['final_recovery']['std']
    print(f"Rate {rate}: {mean:.3f} ± {std:.3f}")
```

---

## Confidence Interval Calculation

The 95% confidence interval uses the t-distribution for small samples:

```python
from scipy import stats

t_value = stats.t.ppf(0.975, n - 1)  # 97.5th percentile
margin = t_value * std / np.sqrt(n)
ci_lower = mean - margin
ci_upper = mean + margin
```

For large n (>30), this converges to approximately ±1.96 standard errors.
