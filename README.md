# Household Recovery Simulation

A RAG-enhanced agent-based model for simulating community disaster recovery. This simulation combines traditional agent-based modeling with Retrieval-Augmented Generation (RAG) to ground behavioral rules in academic research.

## Features

- **Agent-Based Modeling**: Simulate household, infrastructure, and business recovery dynamics
- **RAG-Enhanced Heuristics**: Extract behavioral rules from academic papers via LLM
- **Automatic Parameter Extraction**: Extract numeric parameters (recovery rates, thresholds) from research
- **YAML/JSON Configuration**: Externalize all parameters in config files
- **Multiple Network Topologies**: Barabasi-Albert, Watts-Strogatz, Erdos-Renyi, Random Geometric
- **Monte Carlo Support**: Run multiple simulations with statistical analysis
- **Parameter Merging**: Smart precedence between RAG-extracted, config file, and default values

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd household-recovery

# Install dependencies
pip install -r requirements.txt
```

## Documentation

For detailed documentation, see the `docs/` directory:

- **[Getting Started](docs/getting-started/)** - Installation, quickstart, configuration reference
- **[User Guide](docs/user-guide/)** - Tutorials on simulations, RAG, Monte Carlo, networks
- **[API Reference](docs/api-reference/)** - Complete class and function documentation
- **[Architecture](docs/architecture/)** - System design, data flow, security model
- **[Examples](docs/examples/)** - Runnable Python example scripts

## Quick Start

```bash
# Run with defaults (uses fallback heuristics)
python -m household_recovery

# Run with custom parameters
python -m household_recovery --households 50 --steps 20

# Run with config file
python -m household_recovery --config config.yaml

# Run Monte Carlo simulation
python -m household_recovery --monte-carlo 100 --parallel

# Use local PDFs for heuristic extraction
python -m household_recovery --pdf-dir ~/research/papers --groq-key YOUR_KEY
```

## Configuration

### Using a Config File

Create a YAML configuration file to customize all simulation parameters:

```yaml
# config.yaml
simulation:
  num_households: 50
  num_infrastructure: 3
  num_businesses: 3
  network_type: watts_strogatz
  steps: 20
  base_recovery_rate: 0.08
  utility_weights:
    self_recovery: 1.0
    neighbor_recovery: 0.35
    infrastructure: 0.25
    business: 0.15

thresholds:
  income:
    low: 40000    # Below this = 'low' income
    high: 100000  # Above this = 'high' income
  resilience:
    low: 0.30     # Below this = 'low' resilience
    high: 0.70    # Above this = 'high' resilience

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

Then run with:
```bash
python -m household_recovery --config config.yaml
```

### Parameter Precedence

Parameters are merged with the following precedence (highest to lowest):

1. **RAG-extracted parameters** (if confidence >= threshold, default 0.7)
2. **Config file values**
3. **Hardcoded defaults**

This means:
- Research-grounded values automatically override defaults when confidently extracted
- You can override RAG values by setting explicit values in your config file
- CLI arguments override config file values for basic parameters

### Environment Variables

API keys can be set via environment variables or in a `.env` file:

```bash
export SERPAPI_KEY=your_serpapi_key
export GROQ_API_KEY=your_groq_key
```

## RAG System

The simulation uses academic research to generate behavioral heuristics and extract simulation parameters.

### Heuristic Extraction

The system extracts IF-THEN behavioral rules from paper abstracts:

```
IF neighbors mostly recovered THEN boost recovery by 50%
IF infrastructure is poor AND income is low THEN reduce recovery rate
```

### Parameter Extraction

The enhanced RAG system also extracts numeric parameters:

- **Base recovery rates** from disaster recovery timelines
- **Income thresholds** used in socioeconomic research
- **Resilience scales** from empirical studies
- **Social influence weights** from network analysis

### Data Sources

- **Google Scholar**: Fetch papers via SerpAPI
- **Local PDFs**: Process your own research library
- **Hybrid mode**: Combine both sources

## API Usage

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import ThresholdConfig, FullConfig

# Simple usage with defaults
config = SimulationConfig(num_households=30, steps=15)
engine = SimulationEngine(config)
result = engine.run()

print(f"Final recovery: {result.final_recovery:.3f}")

# With custom thresholds
thresholds = ThresholdConfig(
    income_low=40000,
    income_high=100000,
    resilience_low=0.3,
    resilience_high=0.7,
)

engine = SimulationEngine(
    config=config,
    thresholds=thresholds,
)
result = engine.run()

# From config file
full_config = FullConfig.from_file("config.yaml")
engine = SimulationEngine(
    config=full_config.simulation,
    thresholds=full_config.thresholds,
    infra_config=full_config.infrastructure,
    network_config=full_config.network,
)
```

## Network Topologies

| Topology | Description | Best For |
|----------|-------------|----------|
| `barabasi_albert` | Scale-free network with hubs | Social networks, realistic communities |
| `watts_strogatz` | Small-world network | Information spread analysis |
| `erdos_renyi` | Random network | Baseline comparisons |
| `random_geometric` | Spatial proximity network | Geographic simulations |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=household_recovery --cov-report=html
```

## Project Structure

```
household_recovery/
├── __init__.py          # Package exports
├── __main__.py          # Entry point
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── agents.py            # Agent classes (Household, Infrastructure, Business)
├── network.py           # Network creation and management
├── simulation.py        # Simulation engine and parameter merger
├── heuristics.py        # RAG pipeline and parameter extraction
├── monte_carlo.py       # Multi-run analysis
├── safe_eval.py         # Secure expression evaluation
├── visualization.py     # Plotting and reports
└── pdf_retrieval.py     # Local PDF processing
```

## Key Concepts

### Agent-Based Modeling (ABM)

Agents (households) make autonomous decisions based on:
- Individual characteristics (income, resilience)
- Neighborhood state (neighbor recovery, infrastructure)
- Behavioral heuristics from research

### Utility-Based Decisions

Households evaluate proposed recovery levels using a utility function:

```
utility = w_self * own_recovery
        + w_neighbor * avg_neighbor_recovery
        + w_infra * infrastructure_functionality
        + w_business * business_availability
```

Recovery only happens if utility increases.

### Safe Expression Evaluation

LLM-generated heuristic conditions are validated via AST parsing to prevent code injection. Only whitelisted context keys are allowed in conditions.

## License

MIT License
