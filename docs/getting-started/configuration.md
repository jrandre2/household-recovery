# Configuration Reference

Complete reference for all configuration options.

## Configuration Methods

### 1. Command Line Arguments

```bash
python -m household_recovery --households 50 --steps 20 --network watts_strogatz
```

### 2. YAML/JSON Config File

```bash
python -m household_recovery --config config.yaml
```

### 3. Python API

```python
from household_recovery.config import SimulationConfig, ThresholdConfig, FullConfig
```

## Complete Configuration File

```yaml
# config.yaml - Complete configuration example

simulation:
  num_households: 50          # Number of household agents
  num_infrastructure: 3       # Number of infrastructure nodes
  num_businesses: 3           # Number of business nodes
  network_type: watts_strogatz  # Network topology
  network_connectivity: 4     # Average connections per node
  steps: 20                   # Simulation time steps
  random_seed: 42             # For reproducibility (null for random)
  base_recovery_rate: 0.08    # Base recovery increment per step
  utility_weights:            # Weights for utility function
    self_recovery: 1.0
    neighbor_recovery: 0.35
    infrastructure: 0.25
    business: 0.15

thresholds:
  income:
    low: 40000                # Below = 'low' income
    high: 100000              # Above = 'high' income
  resilience:
    low: 0.30                 # Below = 'low' resilience
    high: 0.70                # Above = 'high' resilience

infrastructure:
  improvement_rate: 0.06      # Base improvement per step
  initial_functionality_min: 0.2
  initial_functionality_max: 0.5
  household_recovery_multiplier: 0.1

network:
  connection_probability: 0.6 # Household-infrastructure connection probability

visualization:
  output_dir: ./output
  save_network_plots: true
  save_progress_plot: true
  figure_dpi: 150             # Use 300 for publication
  show_plots: false

research:
  default_query: "heuristics in agent-based models for community disaster recovery"
  num_papers: 5
  cache_dir: ./.cache/scholar
  cache_expiry_hours: 24

api:
  llm_model: llama-3.3-70b-versatile
  llm_temperature: 0.05
  llm_max_tokens: 1200
```

## Configuration Classes

### SimulationConfig

Core simulation parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_households` | int | 20 | Number of household agents |
| `num_infrastructure` | int | 2 | Number of infrastructure nodes |
| `num_businesses` | int | 2 | Number of business nodes |
| `network_type` | str | 'barabasi_albert' | Network topology |
| `network_connectivity` | int | 2 | Connections per node |
| `steps` | int | 10 | Simulation steps |
| `random_seed` | int/None | None | Seed for reproducibility |
| `base_recovery_rate` | float | 0.1 | Base recovery per step |
| `utility_weights` | dict | see below | Utility function weights |

**Default utility weights:**
```python
{
    'self_recovery': 1.0,
    'neighbor_recovery': 0.3,
    'infrastructure': 0.2,
    'business': 0.2
}
```

### ThresholdConfig

Classification thresholds for agents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `income_low` | float | 45000.0 | Income below this = 'low' |
| `income_high` | float | 120000.0 | Income above this = 'high' |
| `resilience_low` | float | 0.35 | Resilience below this = 'low' |
| `resilience_high` | float | 0.70 | Resilience above this = 'high' |

### InfrastructureConfig

Infrastructure and business parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `improvement_rate` | float | 0.05 | Base improvement per step |
| `initial_functionality_min` | float | 0.2 | Min initial functionality |
| `initial_functionality_max` | float | 0.5 | Max initial functionality |
| `household_recovery_multiplier` | float | 0.1 | Household influence |

### NetworkConfig

Network connection parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_probability` | float | 0.5 | Household-infrastructure connection probability |

### APIConfig

External API settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `serpapi_key` | str | env var | SerpAPI key |
| `groq_api_key` | str | env var | Groq API key |
| `llm_model` | str | 'llama-3.3-70b-versatile' | LLM model |
| `llm_temperature` | float | 0.05 | LLM temperature |
| `llm_max_tokens` | int | 1200 | Max tokens |

### VisualizationConfig

Output and plotting settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | Path | ./output | Output directory |
| `save_network_plots` | bool | True | Save network visualizations |
| `save_progress_plot` | bool | True | Save recovery trajectory |
| `figure_dpi` | int | 150 | Figure resolution |
| `figure_size` | tuple | (12, 9) | Figure dimensions |
| `colormap` | str | 'viridis' | Matplotlib colormap |
| `show_plots` | bool | False | Display interactively |

### ResearchConfig

Paper retrieval settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_query` | str | see code | Search query for papers |
| `num_papers` | int | 5 | Papers to retrieve |
| `cache_dir` | Path | ./.cache/scholar | Cache directory |
| `cache_expiry_hours` | int | 24 | Cache validity |

## Network Types

| Type | Description | Best For |
|------|-------------|----------|
| `barabasi_albert` | Scale-free with hubs | Realistic social networks |
| `watts_strogatz` | Small-world clustering | Information spread |
| `erdos_renyi` | Random connections | Baseline comparisons |
| `random_geometric` | Spatial proximity | Geographic models |

## Parameter Precedence

Parameters are merged with this priority (highest first):

1. **RAG-extracted** - From research papers (if confidence >= 0.7)
2. **Config file** - Values from YAML/JSON
3. **Hardcoded defaults** - Built-in fallbacks

This allows research-grounded values to automatically improve simulations while preserving manual override capability.

## Environment Variables

```bash
# API Keys
export SERPAPI_KEY=your_key
export GROQ_API_KEY=your_key
```

Or use a `.env` file in the project root.

## Loading Configuration in Python

```python
from household_recovery.config import FullConfig

# From file
config = FullConfig.from_file("config.yaml")

# Validate
config.validate()

# Access sections
sim = config.simulation
thresholds = config.thresholds
api = config.api
```

## CLI Arguments Reference

```bash
python -m household_recovery --help
```

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-c` | Path to config file |
| `--households` | `-n` | Number of households |
| `--steps` | `-s` | Simulation steps |
| `--network` | | Network type |
| `--seed` | | Random seed |
| `--output-dir` | `-o` | Output directory |
| `--monte-carlo` | `-m` | Number of Monte Carlo runs |
| `--parallel` | `-p` | Use parallel execution |
| `--pdf-dir` | | Local PDF directory |
| `--serpapi-key` | | SerpAPI key |
| `--groq-key` | | Groq API key |
| `--verbose` | `-v` | Verbose logging |
