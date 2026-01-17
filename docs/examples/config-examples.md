# Configuration Examples

Pre-built YAML configuration files derived from academic research on disaster recovery.

## Overview

The `configs/manual/` directory contains 17 configuration files extracted from peer-reviewed disaster recovery studies. Each file provides empirically-grounded parameter values for the RecovUS decision model.

These configurations allow you to simulate recovery scenarios based on real-world disaster events and research findings.

## Usage

### Command Line

```bash
# Run with a specific configuration
python -m household_recovery --config configs/manual/katrina_stable_housing_2022.yaml

# Override specific parameters
python -m household_recovery --config configs/manual/katrina_stable_housing_2022.yaml --households 100

# Monte Carlo with research-based parameters
python -m household_recovery --config configs/manual/wang_2021_quant_model.yaml --monte-carlo 50
```

### Python API

```python
from household_recovery import SimulationEngine
from household_recovery.config import FullConfig

# Load configuration from YAML
config = FullConfig.from_yaml("configs/manual/katrina_stable_housing_2022.yaml")

# Run simulation
engine = SimulationEngine(
    config=config.simulation,
    recovus_config=config.recovus
)
result = engine.run()
```

---

## Available Configurations

### Hurricane Events

| File | Event | Key Parameters |
|------|-------|----------------|
| `katrina_stable_housing_2022.yaml` | Hurricane Katrina (2005) | r0=0.19, r2=0.66, SBA floor=$20k |
| `hurricane_andrew_1992.yaml` | Hurricane Andrew (1992) | 93% insurance, FEMA max=$5k |
| `hamideh_2018_housing_recovery.yaml` | Hurricane Ike - Galveston | 50% insurance, 20% CDBG-DR |
| `housing_type_matters_ike_2021.yaml` | Hurricane Ike (2008) | Recovery rate=0.033/month |
| `punta_gorda_charley_2004.yaml` | Hurricane Charley (2004) | 92% repair rate, 8% relocation |
| `disaster_disparities_new_orleans_2010.yaml` | Katrina - New Orleans | 72% return rate, 10% buyout |

### Flood Events

| File | Event | Key Parameters |
|------|-------|----------------|
| `wilson_2021.yaml` | General flood recovery | FEMA max=$35.5k, SBA max=$200k |
| `affordable_housing_social_equity.yaml` | Flood - equity focus | FEMA cap=$33k (2018) |
| `levee_risk_perception_2012.yaml` | Levee-protected areas | 20% insurance penetration |
| `lee_2022_evacuation_return.yaml` | Harris County, TX floods | Fast return (0.33/month), 20% lag |
| `frimpong_2025_principal_agent.yaml` | Flood risk management | 57% insurance penetration |

### Wildfire Events

| File | Event | Key Parameters |
|------|-------|----------------|
| `camp_fire_tubbs_2017_2018.yaml` | Camp/Tubbs Fires, CA | 84% insurance, only 9% rebuilt |

### Other Events

| File | Event | Key Parameters |
|------|-------|----------------|
| `west_tx_explosion_2013.yaml` | West, TX explosion | 82% repaired, 11% cleared lots |
| `wang_2021_quant_model.yaml` | Joplin tornado (2011) | 85% insurance, 98.6% full recovery |

### General Recovery Studies

| File | Focus | Notes |
|------|-------|-------|
| `cdbg_dr_2019.yaml` | CDBG-DR funding | 3.8 year average recovery |
| `horney_2016_community_recovery.yaml` | Community indicators | Default parameters |
| `jordan_2013_indicators.yaml` | Recovery indicators | Default parameters |

---

## Parameter Details

### Recovery Rate Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `base_recovery_rate` | Monthly recovery increment | 0.01 - 0.05 |
| `transition_r0` | Probability: waiting -> repairing | 0.15 - 0.40 |
| `transition_r1` | Probability: continue repairing | 0.90 - 0.98 |
| `transition_r2` | Probability: repairing -> recovered | 0.65 - 0.99 |
| `transition_relocate` | Probability: relocate if infeasible | 0.05 - 0.20 |

### Financial Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `fema_ha_max` | Maximum FEMA housing assistance | $5,000 - $35,500 |
| `sba_loan_max` | Maximum SBA disaster loan | $200,000 |
| `sba_income_floor` | Minimum income for SBA consideration | $20,000 - $50,000 |
| `insurance_penetration_rate` | Proportion with insurance | 0.20 - 0.93 |
| `cdbg_dr_coverage_rate` | CDBG-DR funding coverage | 0.10 - 0.20 |

---

## Example: Comparing Scenarios

```python
from household_recovery import SimulationEngine
from household_recovery.config import FullConfig
from household_recovery.monte_carlo import run_monte_carlo

# Compare Katrina vs Andrew recovery dynamics
scenarios = {
    'katrina': 'configs/manual/katrina_stable_housing_2022.yaml',
    'andrew': 'configs/manual/hurricane_andrew_1992.yaml',
}

results = {}
for name, config_path in scenarios.items():
    config = FullConfig.from_yaml(config_path)
    mc_result = run_monte_carlo(
        config=config.simulation,
        recovus_config=config.recovus,
        n_runs=30
    )
    results[name] = mc_result.mean_final_recovery

print(f"Katrina mean recovery: {results['katrina']:.3f}")
print(f"Andrew mean recovery: {results['andrew']:.3f}")
```

---

## Parameter Sources

All parameters are extracted from peer-reviewed academic papers. See `configs/manual/parameters.md` for:
- Direct quotes from source papers
- Derivation notes for calculated values
- Full source citations

---

## See Also

- [Configuration Guide](../getting-started/configuration.md) - Full configuration reference
- [RecovUS User Guide](../user-guide/recovus.md) - Understanding RecovUS parameters
- [RecovUS API Reference](../api-reference/recovus.md) - Technical details
