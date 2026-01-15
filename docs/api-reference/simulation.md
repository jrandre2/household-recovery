# Simulation Module

Core simulation engine and results.

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.simulation import SimulationResult, ParameterMerger
```

## SimulationEngine

Main simulation engine that coordinates all components.

### Constructor

```python
SimulationEngine(
    config: SimulationConfig,
    api_config: APIConfig | None = None,
    research_config: ResearchConfig | None = None,
    heuristics: list[Heuristic] | None = None,
    thresholds: ThresholdConfig | None = None,
    infra_config: InfrastructureConfig | None = None,
    network_config: NetworkConfig | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `SimulationConfig` | required | Simulation configuration |
| `api_config` | `APIConfig` | `None` | API keys for Scholar and LLM |
| `research_config` | `ResearchConfig` | `None` | Paper retrieval configuration |
| `heuristics` | `list[Heuristic]` | `None` | Pre-built heuristics (skips RAG if provided) |
| `thresholds` | `ThresholdConfig` | `None` | Income/resilience classification |
| `infra_config` | `InfrastructureConfig` | `None` | Infrastructure parameters |
| `network_config` | `NetworkConfig` | `None` | Network connection parameters |

### Methods

#### `run(progress_callback=None) -> SimulationResult`

Run the full simulation.

```python
engine = SimulationEngine(config)
result = engine.run()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `progress_callback` | `callable` | Called each step with `(step, avg_recovery)` |

#### `build_knowledge_base() -> list[Heuristic]`

Build or retrieve the knowledge base of heuristics. If heuristics were provided at init, returns those. Otherwise, runs RAG pipeline or uses fallback.

#### `setup_network() -> CommunityNetwork`

Create the community network based on configuration.

### Example

```python
from household_recovery import SimulationEngine, SimulationConfig

config = SimulationConfig(num_households=50, steps=20)
engine = SimulationEngine(config)

# With progress tracking
def on_progress(step, recovery):
    print(f"Step {step}: {recovery:.3f}")

result = engine.run(progress_callback=on_progress)
print(f"Final recovery: {result.final_recovery:.3f}")
```

---

## SimulationResult

Results from a single simulation run.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `SimulationConfig` | Configuration used |
| `recovery_history` | `list[float]` | Recovery levels per step |
| `final_network` | `CommunityNetwork` | Final network state |
| `heuristics_used` | `list[Heuristic]` | Behavioral rules applied |
| `start_time` | `datetime` | When simulation started |
| `end_time` | `datetime` | When simulation ended |
| `random_seed` | `int | None` | Random seed used |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `final_recovery` | `float` | Final average recovery (0-1) |
| `num_steps` | `int` | Number of steps completed |
| `duration_seconds` | `float` | Wall-clock duration |

### Methods

#### `get_household_trajectories() -> dict[int, list[float]]`

Get recovery trajectory for each household.

```python
trajectories = result.get_household_trajectories()
for hh_id, history in trajectories.items():
    print(f"Household {hh_id}: {history[-1]:.3f}")
```

#### `get_final_statistics() -> dict`

Get statistics about the final state.

```python
stats = result.get_final_statistics()
print(f"Mean recovery: {stats['recovery']['mean']:.3f}")
print(f"Std recovery: {stats['recovery']['std']:.3f}")
```

#### `export_csv(filepath: Path | str)`

Export results to CSV for analysis.

```python
result.export_csv("results.csv")
```

#### `export_json(filepath: Path | str)`

Export full results to JSON for reproducibility.

```python
result.export_json("results.json")
```

---

## ParameterMerger

Merges parameters from config file and RAG extraction.

Implements precedence:
1. RAG-extracted (if confidence >= threshold)
2. Config file values
3. Hardcoded defaults

### Constructor

```python
ParameterMerger(
    sim_config: SimulationConfig,
    thresholds: ThresholdConfig,
    extracted: ExtractedParameters | None = None,
    confidence_threshold: float = 0.7
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_base_recovery_rate()` | `tuple[float, str]` | `(value, source)` |
| `get_income_thresholds()` | `tuple[float, float, str]` | `(low, high, source)` |
| `get_resilience_thresholds()` | `tuple[float, float, str]` | `(low, high, source)` |
| `get_utility_weights()` | `tuple[dict, str]` | `(weights, source)` |
| `get_merged_configs()` | `tuple[SimulationConfig, ThresholdConfig]` | Merged configurations |
| `log_merge_decisions()` | `None` | Log all merge decisions |

---

## run_simulation

Convenience function to run a simulation with minimal setup.

```python
from household_recovery.simulation import run_simulation

result = run_simulation(
    steps=15,
    num_households=30,
    seed=42
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `int` | 10 | Simulation steps |
| `num_households` | `int` | 20 | Number of agents |
| `seed` | `int | None` | `None` | Random seed |
| `**kwargs` | | | Additional `SimulationConfig` params |
