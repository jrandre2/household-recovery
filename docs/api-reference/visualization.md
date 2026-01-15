# Visualization Module

Plotting and report generation.

```python
from household_recovery.visualization import (
    apply_publication_style,
    plot_network,
    plot_recovery_trajectory,
    plot_monte_carlo_trajectory,
    plot_recovery_distribution,
    create_simulation_report,
    create_monte_carlo_report
)
```

---

## apply_publication_style

Apply publication-ready matplotlib style.

```python
apply_publication_style()
```

Sets:
- Clean sans-serif fonts
- Appropriate font sizes for labels/titles
- High DPI for saved figures
- Removes top/right spines

---

## plot_network

Visualize the network state with nodes colored by recovery/functionality.

```python
fig = plot_network(
    network=result.final_network,
    step=15,
    title="Recovery Network",
    save_path="network.png",
    show=False,
    figsize=(12, 9),
    dpi=150,
    colormap='viridis'
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | `CommunityNetwork` | required | Network to visualize |
| `step` | `int` | 0 | Current simulation step |
| `title` | `str` | `None` | Custom title |
| `save_path` | `Path | str` | `None` | Path to save figure |
| `show` | `bool` | `False` | Display interactively |
| `figsize` | `tuple[int, int]` | (12, 9) | Figure size |
| `dpi` | `int` | 150 | Resolution |
| `colormap` | `str` | 'viridis' | Matplotlib colormap |

### Node Types

- **Circles**: Households (colored by recovery)
- **Squares**: Infrastructure (colored by functionality)
- **Triangles**: Businesses (colored by availability)

---

## plot_recovery_trajectory

Plot the recovery trajectory from a single simulation.

```python
fig = plot_recovery_trajectory(
    result=simulation_result,
    title="Household Recovery Over Time",
    save_path="trajectory.png",
    show=False,
    figsize=(10, 6),
    dpi=150
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `SimulationResult` | required | Simulation result |
| `title` | `str` | "Household Recovery Over Time" | Plot title |
| `save_path` | `Path | str` | `None` | Save path |
| `show` | `bool` | `False` | Display interactively |
| `figsize` | `tuple[int, int]` | (10, 6) | Figure size |
| `dpi` | `int` | 150 | Resolution |

---

## plot_monte_carlo_trajectory

Plot Monte Carlo results with confidence bands.

```python
fig = plot_monte_carlo_trajectory(
    results=mc_results,
    title="Recovery Trajectory with Confidence Interval",
    save_path="mc_trajectory.png",
    show=False,
    figsize=(10, 6),
    dpi=150,
    show_individual=False,
    ci_alpha=0.3
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `MonteCarloResults` | required | Monte Carlo results |
| `title` | `str` | see above | Plot title |
| `save_path` | `Path | str` | `None` | Save path |
| `show` | `bool` | `False` | Display interactively |
| `figsize` | `tuple[int, int]` | (10, 6) | Figure size |
| `dpi` | `int` | 150 | Resolution |
| `show_individual` | `bool` | `False` | Show individual run trajectories |
| `ci_alpha` | `float` | 0.3 | Transparency of CI band |

### Output

Shows:
- Mean trajectory (solid blue line)
- 95% confidence interval (shaded band)
- Summary statistics box
- Optional: individual run trajectories (faint gray lines)

---

## plot_recovery_distribution

Plot histogram of final recovery values from Monte Carlo runs.

```python
fig = plot_recovery_distribution(
    results=mc_results,
    title="Distribution of Final Recovery",
    save_path="distribution.png",
    show=False,
    figsize=(8, 6),
    dpi=150
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `MonteCarloResults` | required | Monte Carlo results |
| `title` | `str` | "Distribution of Final Recovery" | Plot title |
| `save_path` | `Path | str` | `None` | Save path |
| `show` | `bool` | `False` | Display interactively |
| `figsize` | `tuple[int, int]` | (8, 6) | Figure size |
| `dpi` | `int` | 150 | Resolution |

### Output

Shows:
- Histogram of final recovery values
- Vertical line at mean
- Dashed lines at 95% CI bounds

---

## create_simulation_report

Generate a complete set of visualizations for a simulation.

```python
create_simulation_report(
    result=simulation_result,
    output_dir="./output",
    prefix="simulation"
)
```

### Creates

- `{prefix}_network_final.png` - Network state at final step
- `{prefix}_trajectory.png` - Recovery trajectory plot
- `{prefix}_data.csv` - Results CSV
- `{prefix}_metadata.json` - Full results JSON

---

## create_monte_carlo_report

Generate visualizations for Monte Carlo results.

```python
create_monte_carlo_report(
    results=mc_results,
    output_dir="./output",
    prefix="monte_carlo"
)
```

### Creates

- `{prefix}_trajectory.png` - Trajectory with confidence bands
- `{prefix}_distribution.png` - Distribution histogram
- `{prefix}_summary.csv` - Mean trajectory with CI
- `{prefix}_all_runs.csv` - All individual run trajectories

---

## Publication Style Settings

```python
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}
```

---

## Example Usage

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.visualization import (
    apply_publication_style,
    create_simulation_report,
    create_monte_carlo_report
)

# Apply clean style
apply_publication_style()

# Single run
config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config)
result = engine.run()

create_simulation_report(result, "./output/single_run")

# Monte Carlo
mc_results = run_monte_carlo(config, n_runs=100, parallel=True)
create_monte_carlo_report(mc_results, "./output/monte_carlo")
```
