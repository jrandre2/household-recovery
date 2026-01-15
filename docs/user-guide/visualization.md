# Visualization Guide

Learn how to create publication-ready plots and reports.

## Overview

The visualization module provides:
- Network state visualizations
- Recovery trajectory plots
- Monte Carlo confidence bands
- Distribution histograms
- Automated report generation

## Quick Start

### Single Simulation Report

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.visualization import create_simulation_report

config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config)
result = engine.run()

# Generate complete report
create_simulation_report(result, "./output", prefix="my_sim")
```

Creates:
- `my_sim_trajectory.png` - Recovery over time
- `my_sim_network_final.png` - Final network state
- `my_sim_data.csv` - Tabular data
- `my_sim_metadata.json` - Full results

### Monte Carlo Report

```python
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.visualization import create_monte_carlo_report

results = run_monte_carlo(config, n_runs=100)
create_monte_carlo_report(results, "./output", prefix="mc_study")
```

Creates:
- `mc_study_trajectory.png` - Mean with confidence bands
- `mc_study_distribution.png` - Final recovery histogram
- `mc_study_summary.csv` - Summary statistics
- `mc_study_all_runs.csv` - Individual run data

## Publication Style

Apply consistent, clean styling:

```python
from household_recovery.visualization import apply_publication_style

apply_publication_style()
# All subsequent plots use publication settings
```

Settings applied:
- Clean sans-serif fonts
- Appropriate sizes for labels/titles
- Removed top/right spines
- High DPI (300) for saved figures

## Network Visualization

### Basic Network Plot

```python
from household_recovery.visualization import plot_network

fig = plot_network(
    network=result.final_network,
    step=20,
    title="Community Recovery Network",
    save_path="network.png",
    show=False
)
```

### Customization Options

```python
fig = plot_network(
    network=result.final_network,
    step=20,
    title="Recovery Status",
    save_path="network.png",
    figsize=(14, 10),      # Larger figure
    dpi=300,               # Higher resolution
    colormap='RdYlGn'      # Red-Yellow-Green colormap
)
```

### Understanding the Plot

- **Circles**: Households (colored by recovery 0-1)
- **Squares**: Infrastructure (colored by functionality)
- **Triangles**: Businesses (colored by availability)
- **Edges**: Connections between nodes
- **Colorbar**: Shows state scale (0=poor, 1=good)

## Recovery Trajectory

### Single Run

```python
from household_recovery.visualization import plot_recovery_trajectory

fig = plot_recovery_trajectory(
    result=result,
    title="Household Recovery Over Time",
    save_path="trajectory.png"
)
```

### With Annotations

```python
import matplotlib.pyplot as plt

fig = plot_recovery_trajectory(result, show=True)
plt.axhline(y=0.8, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig("annotated_trajectory.png")
```

## Monte Carlo Visualization

### Trajectory with Confidence Bands

```python
from household_recovery.visualization import plot_monte_carlo_trajectory

fig = plot_monte_carlo_trajectory(
    results=mc_results,
    title="Recovery with 95% Confidence Interval",
    save_path="mc_trajectory.png",
    show_individual=True,  # Show individual runs (faint)
    ci_alpha=0.3          # Transparency of CI band
)
```

### Recovery Distribution

```python
from household_recovery.visualization import plot_recovery_distribution

fig = plot_recovery_distribution(
    results=mc_results,
    title="Distribution of Final Recovery (n=100)",
    save_path="distribution.png"
)
```

Shows:
- Histogram of final recovery values
- Mean (vertical red line)
- 95% CI bounds (dashed lines)

## Custom Plots

### Comparing Scenarios

```python
import matplotlib.pyplot as plt
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.config import SimulationConfig

scenarios = {
    'Low Recovery': 0.05,
    'Medium Recovery': 0.10,
    'High Recovery': 0.15
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, rate in scenarios.items():
    config = SimulationConfig(
        num_households=50,
        steps=20,
        base_recovery_rate=rate
    )
    results = run_monte_carlo(config, n_runs=50)

    ax.plot(results.mean_trajectory, label=name)
    ax.fill_between(
        range(len(results.mean_trajectory)),
        results.ci_lower,
        results.ci_upper,
        alpha=0.2
    )

ax.set_xlabel('Time Step')
ax.set_ylabel('Average Recovery')
ax.set_title('Recovery by Base Rate')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
```

### Household Trajectories

```python
import matplotlib.pyplot as plt

trajectories = result.get_household_trajectories()

fig, ax = plt.subplots(figsize=(10, 6))

for hh_id, history in trajectories.items():
    ax.plot(history, alpha=0.3, linewidth=0.5)

ax.plot(result.recovery_history, 'b-', linewidth=2, label='Mean')
ax.set_xlabel('Time Step')
ax.set_ylabel('Recovery')
ax.set_title('Individual Household Trajectories')
ax.legend()

plt.savefig('household_trajectories.png')
```

### Network Evolution

```python
from household_recovery.visualization import plot_network

# Run simulation step by step and capture network at each step
steps_to_plot = [0, 5, 10, 15, 20]

for step in range(config.steps + 1):
    network.step(heuristics, base_recovery_rate)

    if step in steps_to_plot:
        plot_network(
            network,
            step=step,
            save_path=f"network_step_{step:02d}.png"
        )
```

## Exporting for Publication

### High-Resolution Figures

```python
fig = plot_recovery_trajectory(result)
fig.savefig(
    "figure1.pdf",      # PDF for vector graphics
    dpi=300,            # High resolution
    bbox_inches='tight' # Trim whitespace
)
```

### Figure Size for Journals

Common sizes (width in inches):
- Single column: 3.5"
- 1.5 columns: 5"
- Double column: 7"

```python
fig = plot_recovery_trajectory(
    result,
    figsize=(3.5, 2.5)  # Single column, golden ratio
)
```

## Colormaps

Available colormaps for network plots:

| Colormap | Best For |
|----------|----------|
| `viridis` | General use (default) |
| `RdYlGn` | Red-Yellow-Green (intuitive) |
| `coolwarm` | Diverging data |
| `plasma` | High contrast |

```python
plot_network(network, colormap='RdYlGn')
```

## Interactive Display

For Jupyter notebooks:

```python
%matplotlib inline

from household_recovery.visualization import plot_recovery_trajectory

fig = plot_recovery_trajectory(result, show=True)
# Figure displays inline
```

## Complete Example

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.visualization import (
    apply_publication_style,
    plot_network,
    plot_recovery_trajectory,
    plot_monte_carlo_trajectory,
    plot_recovery_distribution
)
import matplotlib.pyplot as plt

# Apply consistent styling
apply_publication_style()

# Run simulation
config = SimulationConfig(num_households=100, steps=25)
engine = SimulationEngine(config)
result = engine.run()

# Single run visualizations
fig1 = plot_recovery_trajectory(result, save_path="fig1_trajectory.png")
fig2 = plot_network(result.final_network, save_path="fig2_network.png")

# Monte Carlo analysis
mc_results = run_monte_carlo(config, n_runs=100, parallel=True)

fig3 = plot_monte_carlo_trajectory(mc_results, save_path="fig3_mc_trajectory.png")
fig4 = plot_recovery_distribution(mc_results, save_path="fig4_distribution.png")

# Summary figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Re-create plots in subplots for combined figure
# ... (add subplot code)

plt.tight_layout()
plt.savefig("summary_figure.png", dpi=300)

print("Figures saved!")
```

## Next Steps

- [Monte Carlo](monte-carlo.md) - Statistical analysis
- [Basic Simulation](basic-simulation.md) - Understanding outputs
- [Network Topologies](network-topologies.md) - Network visualizations
