# Research-Based Configuration Examples

This project includes 18 pre-built configuration files derived from academic research papers. Each config extracts RecovUS parameters from published studies on disaster recovery.

## Quick Start

```bash
# Use a research-based config
python -m household_recovery --config configs/manual/hurricane_andrew_1992.yaml --households 100

# Combine with disaster funding data
python -m household_recovery --config configs/manual/katrina_stable_housing_2022.yaml \
    --disaster "Hurricane Harvey" --households 100
```

## Available Configurations

### Hurricanes

| Config File | Disaster | Key Parameters | Source |
|-------------|----------|----------------|--------|
| `hurricane_andrew_1992.yaml` | Hurricane Andrew (FL, 1992) | 93% insurance, 7% SBA uptake | Zhang & Peacock |
| `katrina_stable_housing_2022.yaml` | Hurricane Katrina (LA, 2005) | 66% recovery rate, 18.67% r0 | Spader et al. 2022 |
| `hamideh_2018_housing_recovery.yaml` | Hurricane Ike (Galveston, 2008) | 50% insurance, 20% CDBG-DR coverage | Hamideh et al. 2018 |
| `housing_type_matters_ike_2021.yaml` | Hurricane Ike (TX, 2008) | 3.33% monthly recovery rate | Peacock et al. 2021 |
| `punta_gorda_charley_2004.yaml` | Hurricane Charley (FL, 2004) | 92% repair rate, 8% relocation | Comerio 2014 |
| `lee_2022_evacuation_return.yaml` | Hurricane Harvey (Harris County, 2017) | 80% r2, 20% relocate | Lee et al. 2022 |

### Floods

| Config File | Disaster | Key Parameters | Source |
|-------------|----------|----------------|--------|
| `wilson_2021.yaml` | General flood recovery | $35,500 FEMA max, $200k SBA max | Wilson 2021 |
| `frimpong_2025_principal_agent.yaml` | Flood insurance behavior | 57% insurance penetration | Frimpong 2025 |
| `levee_risk_perception_2012.yaml` | Levee-protected areas | 20% insurance penetration | Kousky & Kunreuther 2012 |

### Wildfires

| Config File | Disaster | Key Parameters | Source |
|-------------|----------|----------------|--------|
| `camp_fire_tubbs_2017_2018.yaml` | Camp Fire / Tubbs Fire (CA) | 84% insurance, 9% rebuilt (47mo) | Nejat et al. 2023 |

### Tornadoes

| Config File | Disaster | Key Parameters | Source |
|-------------|----------|----------------|--------|
| `wang_2021_quant_model.yaml` | Joplin Tornado (MO, 2011) | 85% insurance, 98.61% recovery | Wang & Jia 2021 |

### Other Disasters

| Config File | Disaster | Key Parameters | Source |
|-------------|----------|----------------|--------|
| `west_tx_explosion_2013.yaml` | West, TX Explosion (2013) | 82% r2, 11% relocate | Peacock et al. 2018 |
| `disaster_disparities_new_orleans_2010.yaml` | New Orleans (post-Katrina) | 71.9% return rate, 10% buyout | Fussell et al. 2010 |

### General / Multi-Disaster

| Config File | Key Focus | Source |
|-------------|-----------|--------|
| `affordable_housing_social_equity.yaml` | Social equity in recovery | Peacock et al. 2014 |
| `cdbg_dr_2019.yaml` | CDBG-DR recovery timeline | HUD 2019 |
| `horney_2016_community_recovery.yaml` | Recovery indicators | Horney 2016 |
| `jordan_2013_indicators.yaml` | Community recovery metrics | Jordan et al. 2013 |

## Parameter Extraction Details

Each configuration includes parameters extracted directly from research papers with source citations. See [configs/manual/parameters.md](../../configs/manual/parameters.md) for:

- Direct quotes from papers supporting each value
- Derivation notes for calculated parameters
- Source file paths

### Example: Hurricane Andrew (1992)

From `hurricane_andrew_1992.yaml`:

```yaml
recovus:
  insurance_penetration_rate: 0.93
  sba_uptake_rate: 0.07
  fema_ha_max: 5000.0
```

Source quotes from parameters.md:

> "Housing reconstruction ... funded primarily (93%) by insurance settlements"
>
> "supplemental funding (less than 7%) from SBA's loan program"
>
> "FEMA's Minimal Home Repair Program (MHR) provides small grants, usually up to $5,000"

## Comparing Disasters

Use different configs to compare recovery patterns across disaster types:

```bash
# Hurricane with high insurance (Andrew 1992)
python -m household_recovery --config configs/manual/hurricane_andrew_1992.yaml \
    --households 100 --steps 36 --output output/andrew

# Wildfire with slow rebuilding (Camp Fire 2018)
python -m household_recovery --config configs/manual/camp_fire_tubbs_2017_2018.yaml \
    --households 100 --steps 48 --output output/camp_fire

# Tornado with fast recovery (Joplin 2011)
python -m household_recovery --config configs/manual/wang_2021_quant_model.yaml \
    --households 100 --steps 24 --output output/joplin
```

## Monte Carlo with Research Configs

Run Monte Carlo analysis with research-based parameters:

```bash
python -m household_recovery --config configs/manual/katrina_stable_housing_2022.yaml \
    --monte-carlo 100 --parallel --output output/katrina_mc
```

## Creating Custom Research Configs

To create a config from a new research paper:

1. **Extract numeric parameters** from the paper:
   - Recovery rates / timelines
   - Insurance penetration rates
   - FEMA/SBA/CDBG-DR values
   - Relocation rates

2. **Convert to RecovUS format**:
   - `base_recovery_rate`: monthly recovery increment (e.g., 90% in 24mo = 0.0375)
   - `transition_r2`: final recovery rate (e.g., 92% = 0.92)
   - `transition_relocate`: relocation rate (e.g., 8% = 0.08)

3. **Create YAML file**:

```yaml
# configs/manual/your_paper_2024.yaml
simulation:
  steps: 36

recovus:
  enabled: true

  # Financial parameters
  insurance_penetration_rate: 0.75
  fema_ha_max: 35500.0
  sba_uptake_rate: 0.40

  # Transition probabilities
  transition_r2: 0.85
  transition_relocate: 0.10

  # Recovery rate
thresholds:
  income_low: 45000.0
  income_high: 120000.0
```

4. **Document in parameters.md**:

```markdown
- your_paper_2024.yaml (Your Paper Title - Author 2024)
  - insurance_penetration_rate = 0.75
    Quote: "75% of homeowners had active insurance policies"
  - transition_r2 = 0.85
    Quote: "85% of damaged homes were fully repaired within 3 years"
  Source: /path/to/paper.pdf
```

## See Also

- [RecovUS Decision Model](../user-guide/recovus.md) - Understanding parameter meanings
- [Disaster Funding Data](../user-guide/disaster-funding.md) - Using official disaster data
- [Configuration Guide](../getting-started/configuration.md) - Full configuration reference
