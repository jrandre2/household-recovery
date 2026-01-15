# Network Module

Community network management.

```python
from household_recovery.network import CommunityNetwork
```

## CommunityNetwork

The social-infrastructure network for the simulation.

Contains:
- Households (primary agents)
- Infrastructure nodes (power, water, etc.)
- Business nodes (economic entities)
- Graph structure connecting them

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.Graph` | NetworkX graph structure |
| `households` | `dict[int, HouseholdAgent]` | Household agents by ID |
| `infrastructure` | `dict[str, InfrastructureNode]` | Infrastructure nodes by ID |
| `businesses` | `dict[str, BusinessNode]` | Business nodes by ID |
| `rng` | `np.random.Generator` | Random number generator |
| `infra_config` | `InfrastructureConfig` | Infrastructure configuration |

### Class Methods

#### `create(...) -> CommunityNetwork`

Create a new community network.

```python
network = CommunityNetwork.create(
    num_households=50,
    num_infrastructure=3,
    num_businesses=3,
    network_type='watts_strogatz',
    connectivity=4,
    seed=42,
    thresholds=ThresholdConfig(),
    infra_config=InfrastructureConfig(),
    network_config=NetworkConfig()
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_households` | `int` | 20 | Number of households |
| `num_infrastructure` | `int` | 2 | Number of infrastructure nodes |
| `num_businesses` | `int` | 2 | Number of business nodes |
| `network_type` | `NetworkType` | 'barabasi_albert' | Graph topology |
| `connectivity` | `int` | 2 | Connectivity parameter |
| `seed` | `int | None` | `None` | Random seed |
| `thresholds` | `ThresholdConfig` | `None` | Classification thresholds |
| `infra_config` | `InfrastructureConfig` | `None` | Infrastructure params |
| `network_config` | `NetworkConfig` | `None` | Network params |

### Network Types

| Type | Description | Connectivity Meaning |
|------|-------------|---------------------|
| `barabasi_albert` | Scale-free with hubs | Edges per new node |
| `watts_strogatz` | Small-world clustering | Nearest neighbors |
| `erdos_renyi` | Random connections | Expected connections |
| `random_geometric` | Spatial proximity | Radius factor |

### Instance Methods

#### `get_household_neighbors(household_id) -> list[int]`

Get IDs of neighboring households.

```python
neighbors = network.get_household_neighbors(0)
print(f"Household 0 has {len(neighbors)} neighbors")
```

#### `get_connected_infrastructure(household_id) -> list[str]`

Get IDs of infrastructure nodes connected to a household.

```python
infra_ids = network.get_connected_infrastructure(0)
# ['infra_0', 'infra_1']
```

#### `get_connected_businesses(household_id) -> list[str]`

Get IDs of business nodes connected to a household.

```python
bus_ids = network.get_connected_businesses(0)
# ['business_0']
```

#### `get_context_for_household(household_id) -> SimulationContext`

Build the simulation context for a specific household.

```python
context = network.get_context_for_household(0)
print(f"Avg neighbor recovery: {context.avg_neighbor_recovery:.3f}")
print(f"Avg infra functionality: {context.avg_infra_func:.3f}")
```

#### `step(heuristics, base_recovery_rate=0.1, utility_weights=None) -> float`

Execute one simulation step.

1. Each household makes a recovery decision
2. Infrastructure and businesses update based on household recovery

```python
avg_recovery = network.step(
    heuristics=heuristics,
    base_recovery_rate=0.1,
    utility_weights={'self_recovery': 1.0, 'neighbor_recovery': 0.3}
)
```

#### `average_recovery() -> float`

Calculate average household recovery.

```python
avg = network.average_recovery()
print(f"Average recovery: {avg:.3f}")
```

#### `get_statistics() -> dict`

Get current network statistics.

```python
stats = network.get_statistics()

# Returns:
{
    'recovery': {
        'mean': 0.45,
        'std': 0.12,
        'min': 0.2,
        'max': 0.8
    },
    'income': {
        'mean': 75000,
        'std': 35000
    },
    'resilience': {
        'mean': 0.42,
        'std': 0.15
    },
    'network': {
        'num_households': 50,
        'num_edges': 120,
        'avg_degree': 4.8
    }
}
```

### Network Topology Details

#### Barabasi-Albert (Scale-Free)

```
Few highly-connected hubs, many peripheral nodes
- Realistic for social networks
- Robust to random failures
- Vulnerable to targeted attacks on hubs
```

#### Watts-Strogatz (Small-World)

```
High clustering + short path lengths
- "Six degrees of separation" effect
- Information spreads quickly
```

#### Erdos-Renyi (Random)

```
Edges randomly distributed
- Baseline for comparison
- Poisson degree distribution
```

#### Random Geometric

```
Nodes connected based on spatial proximity
- Geographic realism
- Local clustering
```
