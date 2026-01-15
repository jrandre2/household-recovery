# Configuration Module

Configuration dataclasses for the simulation.

```python
from household_recovery.config import (
    SimulationConfig,
    ThresholdConfig,
    InfrastructureConfig,
    NetworkConfig,
    APIConfig,
    VisualizationConfig,
    ResearchConfig,
    FullConfig,
    load_config_file
)
```

## SimulationConfig

Core simulation parameters.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_households` | `int` | 20 | Number of household agents |
| `num_infrastructure` | `int` | 2 | Number of infrastructure nodes |
| `num_businesses` | `int` | 2 | Number of business nodes |
| `network_type` | `NetworkType` | 'barabasi_albert' | Graph topology |
| `network_connectivity` | `int` | 2 | Connections per node |
| `steps` | `int` | 10 | Simulation steps |
| `random_seed` | `int | None` | `None` | Seed for reproducibility |
| `base_recovery_rate` | `float` | 0.1 | Base recovery per step |
| `utility_weights` | `dict[str, float]` | see below | Utility function weights |

### Default Utility Weights

```python
{
    'self_recovery': 1.0,
    'neighbor_recovery': 0.3,
    'infrastructure': 0.2,
    'business': 0.2
}
```

### Methods

#### `copy(**overrides) -> SimulationConfig`

Create a copy with optional overrides.

```python
base = SimulationConfig(num_households=50)
modified = base.copy(steps=30, random_seed=42)
```

---

## ThresholdConfig

Classification thresholds for agents.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `income_low` | `float` | 45000.0 | Below = 'low' income |
| `income_high` | `float` | 120000.0 | Above = 'high' income |
| `resilience_low` | `float` | 0.35 | Below = 'low' resilience |
| `resilience_high` | `float` | 0.70 | Above = 'high' resilience |

### Methods

#### `validate()`

Validate threshold configuration (raises ValueError if invalid).

---

## InfrastructureConfig

Infrastructure and business parameters.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `improvement_rate` | `float` | 0.05 | Base improvement per step |
| `initial_functionality_min` | `float` | 0.2 | Min initial functionality |
| `initial_functionality_max` | `float` | 0.5 | Max initial functionality |
| `household_recovery_multiplier` | `float` | 0.1 | Household influence |

---

## NetworkConfig

Network connection parameters.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_probability` | `float` | 0.5 | Household-infrastructure connection probability |

---

## APIConfig

External API settings.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `serpapi_key` | `str` | env var | SerpAPI key |
| `groq_api_key` | `str` | env var | Groq API key |
| `llm_model` | `str` | 'llama-3.3-70b-versatile' | LLM model |
| `llm_temperature` | `float` | 0.05 | LLM temperature |
| `llm_max_tokens` | `int` | 1200 | Max tokens |

### Methods

#### `validate() -> bool`

Check if required API keys are set.

```python
api_config = APIConfig()
if api_config.validate():
    print("API keys configured")
else:
    print("Using fallback heuristics")
```

---

## VisualizationConfig

Output and plotting settings.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `Path` | ./output | Output directory |
| `save_network_plots` | `bool` | True | Save network visualizations |
| `save_progress_plot` | `bool` | True | Save recovery trajectory |
| `figure_dpi` | `int` | 150 | Figure resolution |
| `figure_size` | `tuple[int, int]` | (12, 9) | Figure dimensions |
| `colormap` | `str` | 'viridis' | Matplotlib colormap |
| `show_plots` | `bool` | False | Display interactively |

---

## ResearchConfig

Paper retrieval settings.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_query` | `str` | see code | Search query |
| `num_papers` | `int` | 5 | Papers to retrieve |
| `cache_dir` | `Path` | ./.cache/scholar | Cache directory |
| `cache_expiry_hours` | `int` | 24 | Cache validity |

---

## FullConfig

Complete configuration combining all components.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `simulation` | `SimulationConfig` | Simulation parameters |
| `visualization` | `VisualizationConfig` | Output settings |
| `research` | `ResearchConfig` | Paper retrieval |
| `api` | `APIConfig` | API keys |
| `thresholds` | `ThresholdConfig` | Classification thresholds |
| `infrastructure` | `InfrastructureConfig` | Infrastructure params |
| `network` | `NetworkConfig` | Network params |

### Class Methods

#### `from_file(filepath) -> FullConfig`

Load configuration from a YAML or JSON file.

```python
config = FullConfig.from_file("config.yaml")
```

#### `from_dict(data) -> FullConfig`

Create config from dictionary.

```python
config = FullConfig.from_dict({
    'simulation': {'num_households': 50},
    'thresholds': {
        'income': {'low': 40000, 'high': 100000}
    }
})
```

### Instance Methods

#### `validate()`

Validate all configuration sections.

#### `to_dict() -> dict`

Convert configuration to dictionary.

---

## load_config_file

Load configuration from YAML or JSON.

```python
data = load_config_file("config.yaml")
# Returns dict
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `Path | str` | Path to config file (.yaml, .yml, or .json) |

---

## Network Type

Type alias for supported network topologies.

```python
NetworkType = Literal[
    'barabasi_albert',
    'watts_strogatz',
    'erdos_renyi',
    'random_geometric'
]
```

---

## Example Configuration File

```yaml
simulation:
  num_households: 50
  num_infrastructure: 3
  num_businesses: 3
  network_type: watts_strogatz
  network_connectivity: 4
  steps: 20
  base_recovery_rate: 0.08
  utility_weights:
    self_recovery: 1.0
    neighbor_recovery: 0.35
    infrastructure: 0.25
    business: 0.15

thresholds:
  income:
    low: 40000
    high: 100000
  resilience:
    low: 0.30
    high: 0.70

infrastructure:
  improvement_rate: 0.06
  initial_functionality_min: 0.2
  initial_functionality_max: 0.5

network:
  connection_probability: 0.6

api:
  llm_model: llama-3.3-70b-versatile
  llm_temperature: 0.05
```
