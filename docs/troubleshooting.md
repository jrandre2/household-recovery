# Troubleshooting Guide

This guide covers common issues and their solutions when working with the Household Recovery Simulation.

## Installation Issues

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'networkx'`

**Solution:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install networkx numpy matplotlib pyyaml
```

### Python Version Errors

**Problem:** `SyntaxError: invalid syntax` on type hints

**Solution:** This package requires Python 3.10+. Check your version:
```bash
python --version
```

If using an older version, upgrade or use a virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Import Errors After Installation

**Problem:** `ImportError: cannot import name 'SimulationEngine'`

**Solution:** Ensure you're running from the project root directory:
```bash
cd /path/to/household-recovery
python -m household_recovery
```

Or install as a package:
```bash
pip install -e .
```

---

## API Key Configuration

### SERPAPI_KEY Not Set

**Problem:**
```
WARNING: SERPAPI_KEY not set - will use fallback heuristics
```

**Solutions:**

1. Set environment variable:
```bash
export SERPAPI_KEY=your_key_here
```

2. Create a `.env` file in the project root:
```
SERPAPI_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

3. Pass directly to functions:
```python
from household_recovery import build_knowledge_base

heuristics = build_knowledge_base(
    serpapi_key="your_key_here",
    groq_api_key="your_key_here"
)
```

**Note:** Without API keys, the simulation uses fallback heuristics from `get_fallback_heuristics()`. This is valid for testing but may not reflect research-grounded behavior.

### GROQ_API_KEY Not Set

**Problem:**
```
WARNING: GROQ_API_KEY not set - will use fallback heuristics
```

**Solution:** Same as SERPAPI_KEY - set via environment variable, `.env` file, or pass directly.

### Invalid API Key

**Problem:**
```
groq.AuthenticationError: Invalid API key
```

**Solution:**
1. Verify your API key is correct (no extra spaces)
2. Check the key hasn't expired
3. Ensure you're using the correct key type (Groq, not OpenAI)

---

## RAG Pipeline Issues

### No Heuristics Extracted

**Problem:** `build_knowledge_base()` returns empty list

**Possible Causes:**

1. **API keys not set** - Use fallback heuristics:
```python
from household_recovery import get_fallback_heuristics
heuristics = get_fallback_heuristics()
```

2. **No relevant papers found** - Try a different query:
```python
# More specific query
heuristics = build_knowledge_base(
    query="household financial recovery hurricane flood damage",
    # ...
)
```

3. **US-only filter excluded papers** - Disable if needed:
```python
from household_recovery.config import ResearchConfig
research_config = ResearchConfig(us_only=False)
```

4. **LLM failed to extract** - Check Groq API status or lower temperature:
```python
from household_recovery.config import APIConfig
api_config = APIConfig(llm_temperature=0.01)
```

### PDF Extraction Fails

**Problem:** `build_knowledge_base_from_pdfs()` returns empty results

**Solutions:**

1. Check PDF directory exists and contains PDFs:
```python
from pathlib import Path
pdf_dir = Path("~/research/papers").expanduser()
print(list(pdf_dir.glob("*.pdf")))
```

2. Ensure PDFs are text-based (not scanned images)

3. If you're filtering to US-only, confirm the papers are US-based (or disable the filter)

4. Try with verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### RecovUS Parameters Not Applied

**Problem:** RAG extraction runs but RecovUS parameters aren't applied

**Solutions:**

1. Check confidence threshold:
```python
result = build_full_knowledge_base(...)
if result.has_recovus_params():
    # Lower threshold if needed (default is 0.7)
    config = result.recovus_params.apply_to_config(
        base_config,
        confidence_threshold=0.5  # Lower threshold
    )
```

2. Check what was extracted:
```python
print(result.summary())
# Shows confidence scores for each parameter
```

---

## RecovUS Configuration Errors

### Perception Distribution Doesn't Sum to 1.0

**Problem:**
```
ValueError: Perception percentages must sum to 1.0, got 0.95
```

**Solution:** Ensure the three perception values sum to exactly 1.0:
```python
from household_recovery.config import RecovUSConfig

config = RecovUSConfig(
    perception_infrastructure=0.65,
    perception_social=0.31,
    perception_community=0.04,  # 0.65 + 0.31 + 0.04 = 1.0
)
```

### Invalid Probability Value

**Problem:**
```
ValueError: transition_r0 must be in [0, 1], got 1.5
```

**Solution:** All probabilities must be between 0.0 and 1.0:
```python
config = RecovUSConfig(
    transition_r0=0.35,   # Valid: 0.0 - 1.0
    transition_r1=0.95,
    transition_r2=0.95,
)
```

### Invalid Adequacy Threshold

**Problem:**
```
ValueError: adequacy_infrastructure must be in [0, 1], got -0.1
```

**Solution:** Adequacy thresholds must be between 0.0 and 1.0:
```python
config = RecovUSConfig(
    adequacy_infrastructure=0.50,  # 50% threshold
    adequacy_neighbor=0.40,
    adequacy_community_assets=0.50,
)
```

---

## Simulation Runtime Issues

### Simulation Runs Very Slowly

**Possible Causes and Solutions:**

1. **Too many households** - Reduce for initial testing:
```python
config = SimulationConfig(num_households=50)  # Start small
```

2. **Too many heuristics** - Each heuristic is evaluated per household per step:
```python
# Limit heuristics
heuristics = heuristics[:10]  # Use only first 10
```

3. **Network too dense** - Use sparser network:
```python
config = SimulationConfig(
    network_type='barabasi_albert',
    network_connectivity=2,  # Lower = fewer connections
)
```

### Out of Memory During Monte Carlo

**Problem:** Python crashes during Monte Carlo with many runs

**Solutions:**

1. Reduce parallel workers:
```python
from household_recovery import run_monte_carlo

results = run_monte_carlo(
    config,
    num_runs=100,
    parallel=True,
    max_workers=2  # Reduce from default
)
```

2. Run sequentially for large simulations:
```python
results = run_monte_carlo(config, num_runs=100, parallel=False)
```

3. Reduce simulation size:
```python
config = SimulationConfig(
    num_households=30,  # Smaller
    steps=15,
)
```

### Random Results Between Runs

**Problem:** Different results each time the simulation runs

**Solution:** Set a random seed for reproducibility:
```python
config = SimulationConfig(
    num_households=50,
    steps=20,
    random_seed=42,  # Fixed seed
)
```

---

## Network Topology Issues

### Invalid Network Type

**Problem:**
```
ValueError: Unknown network type: 'scale_free'
```

**Solution:** Use valid network type names:
```python
# Valid types:
config = SimulationConfig(network_type='barabasi_albert')  # Scale-free
config = SimulationConfig(network_type='watts_strogatz')   # Small-world
config = SimulationConfig(network_type='erdos_renyi')      # Random
config = SimulationConfig(network_type='random_geometric') # Spatial
```

### Disconnected Network

**Problem:** Some households have no neighbors

**Solutions:**

1. Increase connectivity parameter:
```python
config = SimulationConfig(
    network_type='barabasi_albert',
    network_connectivity=3,  # Increase from 2
)
```

2. Use a different network type:
```python
config = SimulationConfig(
    network_type='watts_strogatz',  # More uniform connectivity
    network_connectivity=4,
)
```

---

## Configuration File Issues

### YAML Parse Error

**Problem:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution:** Check YAML syntax - common issues:

```yaml
# WRONG: Missing space after colon
num_households:50

# CORRECT:
num_households: 50
```

```yaml
# WRONG: Tab characters
simulation:
	num_households: 50

# CORRECT: Use spaces
simulation:
  num_households: 50
```

### Config File Not Found

**Problem:**
```
FileNotFoundError: Config file not found: config.yaml
```

**Solution:**
1. Check the file path is correct
2. Use absolute path if needed:
```bash
python -m household_recovery --config /full/path/to/config.yaml
```

### Unknown Configuration Key

**Problem:**
```
TypeError: __init__() got an unexpected keyword argument 'num_agents'
```

**Solution:** Check the correct parameter name:
```yaml
# WRONG:
simulation:
  num_agents: 50

# CORRECT:
simulation:
  num_households: 50
```

---

## Heuristic Evaluation Errors

### Invalid Heuristic Condition

**Problem:**
```
ValueError: Unsafe expression: import os
```

**Solution:** Heuristics are sanitized for security. Only these context keys are allowed:
- `income_level`, `resilience_level`, `recovery`
- `avg_neighbor_recovery`, `avg_infra_func`, `avg_business_avail`
- `perception_type`, `damage_severity`, `recovery_state`
- `is_feasible`, `is_adequate`, `is_habitable`

### Heuristic Not Firing

**Problem:** Heuristic condition never matches

**Debugging:**
```python
# Print context to see available values
def debug_heuristic(heuristic, context):
    print(f"Condition: {heuristic['condition']}")
    print(f"Context: {context}")

    from household_recovery.safe_eval import safe_eval
    result = safe_eval(heuristic['condition'], context)
    print(f"Result: {result}")
```

Common issues:
- String comparisons need quotes: `income_level == 'low'` not `income_level == low`
- Numeric comparisons: `recovery > 0.5` not `recovery > '0.5'`

---

## Visualization Issues

### Plots Not Showing

**Problem:** `engine.run()` completes but no plots appear

**Solution:** Enable plot display in config:
```python
from household_recovery.config import VisualizationConfig

viz_config = VisualizationConfig(show_plots=True)
```

Or save plots to file:
```python
viz_config = VisualizationConfig(
    save_network_plots=True,
    save_progress_plot=True,
    output_dir=Path("./output"),
)
```

### Matplotlib Backend Error

**Problem:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solution:**
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'MacOSX'
import matplotlib.pyplot as plt
```

Or set in config file:
```yaml
visualization:
  show_plots: false
  save_progress_plot: true
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs** - Run with debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Search existing issues** - https://github.com/anthropics/household-recovery/issues

3. **Open a new issue** with:
   - Python version (`python --version`)
   - Package versions (`pip freeze`)
   - Full error traceback
   - Minimal code to reproduce

---

## Related Documentation

- [Installation Guide](getting-started/installation.md)
- [Configuration Reference](getting-started/configuration.md)
- [RecovUS Guide](user-guide/recovus.md)
- [API Reference](api-reference/index.md)
