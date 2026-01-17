# Disaster Funding Module

Data structures and utilities for disaster-specific funding data integration.

```python
from household_recovery.disaster_funding import (
    DisasterFundingRecord,
    TranslatedParameters,
    FundingParameterTranslator,
    DisasterFundingRegistry,
    load_disaster_record,
)
```

---

## DisasterFundingRecord

Dataclass storing official funding data for a specific disaster event.

### Attributes

#### Disaster Identification

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `disaster_name` | `str` | required | Name (e.g., "Hurricane Harvey") |
| `disaster_number` | `int \| None` | `None` | FEMA disaster number (e.g., 4332) |
| `disaster_type` | `str` | "hurricane" | Type: hurricane, flood, wildfire, tornado |
| `declaration_date` | `str \| None` | `None` | ISO date string |
| `state` | `str \| None` | `None` | Primary affected state |

#### Affected Population

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `affected_households` | `int \| None` | `None` | Total affected households |
| `eligible_households` | `int \| None` | `None` | Households eligible for assistance |

#### CDBG-DR Funding

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cdbg_dr_total_allocation` | `float \| None` | `None` | Total $ allocated |
| `cdbg_dr_housing_allocation` | `float \| None` | `None` | Housing-specific portion |
| `cdbg_dr_households_served` | `int \| None` | `None` | Number of households served |
| `cdbg_dr_average_award` | `float \| None` | `None` | Average award per household |

#### FEMA Individual Assistance

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `fema_ia_total_approved` | `float \| None` | `None` | Total IA approved |
| `fema_ha_total` | `float \| None` | `None` | Housing Assistance portion |
| `fema_ha_average` | `float \| None` | `None` | Average HA grant |
| `fema_ha_max_at_time` | `float \| None` | `None` | HA cap at time of disaster |
| `fema_registrations` | `int \| None` | `None` | Total registrations |
| `fema_approvals` | `int \| None` | `None` | Approved registrations |

#### SBA Disaster Loans

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `sba_loans_approved` | `int \| None` | `None` | Number of loans approved |
| `sba_total_amount` | `float \| None` | `None` | Total loan amount |
| `sba_average_loan` | `float \| None` | `None` | Average loan amount |
| `sba_approval_rate` | `float \| None` | `None` | Approved / Applied ratio |

#### NFIP Claims

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `nfip_claims_count` | `int \| None` | `None` | Number of claims |
| `nfip_total_paid` | `float \| None` | `None` | Total claims paid |
| `nfip_average_claim` | `float \| None` | `None` | Average claim amount |
| `nfip_policies_in_force` | `int \| None` | `None` | Policies in force (for penetration rate) |

#### Insurance

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `private_insurance_claims` | `int \| None` | `None` | Private insurance claims |
| `private_insurance_total` | `float \| None` | `None` | Private insurance total |
| `insurance_penetration_rate` | `float \| None` | `None` | Override penetration rate |

#### Recovery Outcomes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `recovery_rate_1yr` | `float \| None` | `None` | Recovery rate at 1 year |
| `recovery_rate_3yr` | `float \| None` | `None` | Recovery rate at 3 years |
| `relocation_rate` | `float \| None` | `None` | Permanent relocation rate |

#### Metadata

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_sources` | `list[str]` | `[]` | List of data source citations |
| `last_updated` | `str \| None` | `None` | Last update date |
| `notes` | `str \| None` | `None` | Additional notes |

### Class Methods

#### `from_yaml(filepath: Path | str) -> DisasterFundingRecord`

Load a disaster funding record from a YAML file.

```python
record = DisasterFundingRecord.from_yaml("data/disasters/harvey.yaml")
print(record.disaster_name)  # "Hurricane Harvey"
```

#### `from_dict(data: dict) -> DisasterFundingRecord`

Create a record from a dictionary.

```python
record = DisasterFundingRecord.from_dict({
    "disaster_name": "Hurricane Harvey",
    "disaster_number": 4332,
    "affected_households": 895000,
})
```

### Instance Methods

#### `to_dict() -> dict`

Convert record to dictionary for serialization.

---

## TranslatedParameters

Dataclass containing parameters translated from disaster funding data.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `insurance_penetration_rate` | `float \| None` | `None` | Translated insurance rate |
| `fema_ha_max` | `float \| None` | `None` | Translated FEMA HA max |
| `sba_uptake_rate` | `float \| None` | `None` | Translated SBA uptake rate |
| `cdbg_dr_probability` | `float \| None` | `None` | Translated CDBG-DR probability |
| `cdbg_dr_coverage_rate` | `float \| None` | `None` | Translated CDBG-DR coverage |
| `sources` | `dict[str, str]` | `{}` | Source attribution for each parameter |

### Methods

#### `to_dict() -> dict`

Convert to dictionary including sources.

---

## FundingParameterTranslator

Translates aggregate disaster funding data to simulation parameters.

### Class Attributes

| Attribute | Value | Description |
|-----------|-------|-------------|
| `LOW_INCOME_FRACTION` | 0.30 | Assumed fraction of eligible households that are low-income |

### Methods

#### `translate(funding_record: DisasterFundingRecord) -> TranslatedParameters`

Translate a funding record to simulation parameters.

```python
translator = FundingParameterTranslator()
record = DisasterFundingRecord.from_yaml("data/disasters/harvey.yaml")
params = translator.translate(record)

print(params.insurance_penetration_rate)  # 0.56
print(params.sources["insurance_penetration_rate"])  # "Hurricane Harvey NFIP policies / affected HH"
```

#### `apply_to_config(funding_record, base_config=None) -> tuple[RecovUSConfig, list]`

Apply funding data to create a modified RecovUSConfig.

```python
translator = FundingParameterTranslator()
record = DisasterFundingRecord.from_yaml("data/disasters/harvey.yaml")

config, merge_log = translator.apply_to_config(record)

for param, value, source in merge_log:
    print(f"{param}: {value} (from {source})")
```

Returns:
- `RecovUSConfig`: Modified configuration with funding data applied
- `list[tuple[str, Any, str]]`: Merge log with (parameter, value, source) tuples

---

## DisasterFundingRegistry

Registry for managing multiple disaster funding records.

### Methods

#### `add(record: DisasterFundingRecord) -> None`

Add a disaster record to the registry.

```python
registry = DisasterFundingRegistry()
registry.add(record)
```

#### `get(disaster_name: str) -> DisasterFundingRecord | None`

Look up a disaster by name (case-insensitive).

```python
record = registry.get("Hurricane Harvey")
```

#### `get_by_number(disaster_number: int) -> DisasterFundingRecord | None`

Look up a disaster by FEMA disaster number.

```python
record = registry.get_by_number(4332)
```

#### `list_disasters(disaster_type: str | None = None) -> list[str]`

List all registered disasters, optionally filtered by type.

```python
all_disasters = registry.list_disasters()
hurricanes = registry.list_disasters("hurricane")
```

### Class Methods

#### `load_from_directory(directory: Path | str) -> DisasterFundingRegistry`

Load all YAML files from a directory into a registry.

```python
registry = DisasterFundingRegistry.load_from_directory("data/disasters")
```

#### `load_builtin() -> DisasterFundingRegistry`

Load the built-in registry of known disasters from `data/disasters/`.

```python
registry = DisasterFundingRegistry.load_builtin()
disasters = registry.list_disasters()
```

---

## Helper Functions

### `load_disaster_record(...) -> DisasterFundingRecord | None`

Load a disaster funding record by name, number, or file path.

```python
from household_recovery.disaster_funding import load_disaster_record

# By name
record = load_disaster_record(disaster_name="Hurricane Harvey")

# By FEMA number
record = load_disaster_record(disaster_number=4332)

# From file
record = load_disaster_record(disaster_file="path/to/disaster.yaml")
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `disaster_name` | `str \| None` | Disaster name to look up |
| `disaster_number` | `int \| None` | FEMA disaster number |
| `disaster_file` | `Path \| str \| None` | Path to YAML file |

#### Returns

`DisasterFundingRecord | None`: The loaded record, or `None` if not found.

#### Priority

1. Explicit file path (if provided)
2. Built-in registry lookup by name
3. Built-in registry lookup by number

---

## Example Usage

### Complete Workflow

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.config import DisasterFundingConfig, RecovUSConfig
from household_recovery.disaster_funding import (
    DisasterFundingRegistry,
    FundingParameterTranslator,
    load_disaster_record,
)

# Option 1: Use high-level config
funding_config = DisasterFundingConfig(disaster_name="Hurricane Harvey")
engine = SimulationEngine(
    SimulationConfig(num_households=100, steps=24),
    disaster_funding_config=funding_config,
)
result = engine.run()

# Option 2: Manual translation
record = load_disaster_record(disaster_name="Hurricane Harvey")
translator = FundingParameterTranslator()
recovus_config, merge_log = translator.apply_to_config(record)

engine = SimulationEngine(
    SimulationConfig(num_households=100, steps=24),
    recovus_config=recovus_config,
)
result = engine.run()
```

### Custom Disaster Registry

```python
from household_recovery.disaster_funding import (
    DisasterFundingRegistry,
    DisasterFundingRecord,
)

# Create custom registry
registry = DisasterFundingRegistry()

# Add custom disaster
my_disaster = DisasterFundingRecord(
    disaster_name="Local Flood 2024",
    disaster_type="flood",
    affected_households=5000,
    eligible_households=4000,
    insurance_penetration_rate=0.45,
    cdbg_dr_probability=0.25,
)
registry.add(my_disaster)

# Look up
record = registry.get("Local Flood 2024")
```
