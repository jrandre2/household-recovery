# Disaster-Specific Funding Data

This guide explains how to use known disaster-specific funding data to calibrate your simulations with real-world values.

## Why Use Disaster-Specific Data?

By default, the simulation uses generic funding parameters (e.g., 60% insurance penetration, 30% CDBG-DR probability). However, **actual disasters have known funding outcomes** that can make your simulations more accurate:

- **Hurricane Harvey (2017)**: $5.024B CDBG-DR, 56% insurance penetration
- **Hurricane Katrina (2005)**: Different funding patterns and recovery outcomes
- **Camp Fire (2018)**: Wildfire-specific insurance and assistance patterns

Using official data allows you to:

1. **Calibrate simulations** to specific historical disasters
2. **Validate model outputs** against known recovery outcomes
3. **Compare scenarios** across different disaster types

## Quick Start

### List Available Disasters

```bash
python -m household_recovery --list-disasters
```

Output:
```
Available disasters in registry:
  - Hurricane Harvey (DR-4332)
```

### Run with Disaster Data

```bash
# By name
python -m household_recovery --disaster "Hurricane Harvey" --households 100 --steps 24

# By FEMA disaster number
python -m household_recovery --disaster-number 4332 --households 100 --steps 24

# From custom YAML file
python -m household_recovery --disaster-file path/to/your_disaster.yaml
```

### Python API

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import DisasterFundingConfig

# Configure disaster funding
funding_config = DisasterFundingConfig(
    disaster_name="Hurricane Harvey"
)

# Run simulation
config = SimulationConfig(num_households=100, steps=24)
engine = SimulationEngine(
    config,
    disaster_funding_config=funding_config
)
result = engine.run()
```

## How Official Data Translates to Parameters

When you specify a disaster, the system automatically translates aggregate funding data into simulation parameters:

| Official Data | Simulation Parameter | Formula |
|---------------|---------------------|---------|
| NFIP policies / affected households | `insurance_penetration_rate` | policies / households |
| FEMA HA cap at time of disaster | `fema_ha_max` | Direct value |
| SBA approval rate | `sba_uptake_rate` | approved / applied |
| CDBG-DR households served / eligible | `cdbg_dr_probability` | served / (eligible × 0.30) |
| CDBG-DR average award / home value | `cdbg_dr_coverage_rate` | avg_award / est_home_value |

### Example: Hurricane Harvey Translation

From official Harvey data:

```yaml
affected_households: 895000
nfip_policies_in_force: 500000
fema_ha_max_at_time: 33000
sba_approval_rate: 0.42
cdbg_dr_households_served: 40000
eligible_households: 750000
cdbg_dr_average_award: 87500
```

Translated to simulation parameters:

```
insurance_penetration_rate: 0.56    (500k / 895k)
fema_ha_max: 33000                  (cap at time)
sba_uptake_rate: 0.42               (actual approval rate)
cdbg_dr_probability: 0.18           (40k / (750k × 0.30))
cdbg_dr_coverage_rate: 0.35         ($87.5k / $250k est. home)
```

## Creating Custom Disaster Files

You can create YAML files for any disaster using official data sources.

### Data Sources

| Source | Data Available | Where to Find |
|--------|---------------|---------------|
| OpenFEMA | FEMA IA, NFIP claims, declarations | [fema.gov/openfema](https://www.fema.gov/openfema) |
| HUD Exchange | CDBG-DR allocations | Federal Register notices |
| SBA | Disaster loan statistics | SBA disaster reports |

### YAML File Template

```yaml
# data/disasters/your_disaster.yaml
disaster_name: "Your Disaster Name"
disaster_number: 1234              # FEMA disaster number
disaster_type: hurricane           # hurricane, flood, wildfire, tornado, etc.
declaration_date: "2024-01-15"
state: TX

# Affected population (required for translation)
affected_households: 100000
eligible_households: 80000

# CDBG-DR funding
cdbg_dr_total_allocation: 1000000000
cdbg_dr_households_served: 10000
cdbg_dr_average_award: 50000

# FEMA Individual Assistance
fema_ha_total: 500000000
fema_ha_average: 5000
fema_ha_max_at_time: 35500
fema_registrations: 100000
fema_approvals: 40000

# SBA Disaster Loans
sba_loans_approved: 5000
sba_total_amount: 300000000
sba_approval_rate: 0.35

# NFIP Claims
nfip_policies_in_force: 50000
nfip_claims_count: 20000

# Optional: Override insurance penetration directly
insurance_penetration_rate: 0.50

# Recovery outcomes (for validation)
recovery_rate_1yr: 0.40
recovery_rate_3yr: 0.80
relocation_rate: 0.05

# Metadata
data_sources:
  - "OpenFEMA API"
  - "HUD Federal Register Notice"
last_updated: "2024-01-15"
```

### Save Location

Place custom disaster files in:
- `data/disasters/` directory (for built-in registry lookup)
- Any location (use `--disaster-file` to specify path)

## Configuration Options

### DisasterFundingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_builtin_registry` | bool | True | Load disasters from `data/disasters/` |
| `disaster_name` | str | None | Disaster name to load (e.g., "Hurricane Harvey") |
| `disaster_number` | int | None | FEMA disaster number (e.g., 4332) |
| `disaster_file` | Path | None | Path to custom YAML file |
| `prefer_official_over_research` | bool | True | Official data overrides RAG-extracted values |

### YAML Configuration

```yaml
# In your config.yaml
disaster_funding:
  use_builtin_registry: true
  disaster_name: "Hurricane Harvey"
  prefer_official_over_research: true
```

## Precedence Rules

When disaster funding data is provided, parameters are merged with this precedence (highest to lowest):

1. **Official disaster funding data** (from YAML)
2. **RAG-extracted values** (from research papers, if confidence >= 0.7)
3. **Config file values**
4. **Hardcoded defaults**

This means official disaster data will override values extracted from research papers, ensuring your simulation uses verified numbers.

## Example Output

When running with disaster data, you'll see the applied parameters in the log:

```
INFO: ============================================================
INFO: Household Recovery Simulation
INFO: ============================================================
INFO: Households: 100
INFO: Steps: 24
INFO: Disaster data: Hurricane Harvey
INFO: ============================================================
INFO: Loaded disaster funding data: Hurricane Harvey
INFO:   insurance_penetration_rate: 0.56 (from Hurricane Harvey (explicit))
INFO:   fema_ha_max: 33000 (from Hurricane Harvey (cap at time))
INFO:   sba_uptake_rate: 0.42 (from Hurricane Harvey SBA approval rate)
INFO:   cdbg_dr_probability: 0.18 (from Hurricane Harvey CDBG-DR households)
INFO:   cdbg_dr_coverage_rate: 0.35 (from Hurricane Harvey avg award)
```

## Validation with Known Outcomes

Hurricane Harvey data includes known recovery outcomes that you can use to validate your simulation:

```yaml
recovery_rate_1yr: 0.35   # ~35% recovered at 1 year
recovery_rate_3yr: 0.75   # ~75% recovered at 3 years
relocation_rate: 0.08     # ~8% permanently relocated
```

Compare your simulation's final recovery rate and relocation patterns against these benchmarks to validate model calibration.

## Next Steps

- See [Research Configurations](../examples/research-configs.md) for 18 pre-built configs from academic papers
- See [RecovUS Decision Model](recovus.md) for details on the financial feasibility model
- See [API Reference](../api-reference/disaster-funding.md) for programmatic access
