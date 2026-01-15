# Network Topologies Guide

Learn how different network structures affect simulation outcomes.

## Overview

Network topology significantly affects how households recover. The connections between households, infrastructure, and businesses determine:
- How information and influence spread
- Which households have access to resources
- How resilient the community is to disruptions

## Available Network Types

### Barabasi-Albert (Scale-Free)

```yaml
network_type: barabasi_albert
network_connectivity: 2  # Edges per new node
```

**Properties:**
- Few highly-connected "hub" households
- Many loosely-connected households
- Power-law degree distribution

**Best for:**
- Modeling realistic social networks
- Studying influence of key community members
- Understanding vulnerability to targeted disruption

**Characteristics:**
- Hubs can accelerate or slow community recovery
- Robust to random household removal
- Vulnerable if hubs are affected

### Watts-Strogatz (Small-World)

```yaml
network_type: watts_strogatz
network_connectivity: 4  # Neighbors in regular lattice
```

**Properties:**
- High clustering (neighbors know each other)
- Short path lengths ("six degrees of separation")
- Balance between local and global connections

**Best for:**
- Information spread studies
- Community-based interventions
- Realistic neighborhood structures

**Characteristics:**
- Fast recovery spread once started
- Local clusters can recover together
- Long-range connections bridge communities

### Erdos-Renyi (Random)

```yaml
network_type: erdos_renyi
network_connectivity: 3  # Expected connections
```

**Properties:**
- Edges randomly distributed
- No hubs or clustering
- Poisson degree distribution

**Best for:**
- Baseline comparisons
- Null model for network effects
- Understanding pure random mixing

**Characteristics:**
- Homogeneous connectivity
- No structural advantages
- Recovery depends mainly on individual factors

### Random Geometric

```yaml
network_type: random_geometric
network_connectivity: 3  # Radius factor
```

**Properties:**
- Nodes connected based on spatial proximity
- Local clustering
- Geographic realism

**Best for:**
- Geographic disaster studies
- Physical infrastructure modeling
- Spatially-constrained recovery

**Characteristics:**
- Neighbors are literally neighbors
- Geographic barriers affect connectivity
- Local recovery clusters

## Choosing a Network Type

| If studying... | Use |
|----------------|-----|
| Social influence | Barabasi-Albert |
| Information spread | Watts-Strogatz |
| Baseline comparison | Erdos-Renyi |
| Geographic effects | Random Geometric |

## Connectivity Parameter

The `network_connectivity` parameter means different things for each type:

| Type | Parameter Meaning |
|------|-------------------|
| Barabasi-Albert | Edges added per new node (m) |
| Watts-Strogatz | Neighbors in initial ring (k/2) |
| Erdos-Renyi | Expected average degree |
| Random Geometric | Connection radius factor |

### Effect of Connectivity

```python
# Low connectivity - sparse network
config = SimulationConfig(network_connectivity=2)

# Medium connectivity
config = SimulationConfig(network_connectivity=4)

# High connectivity - dense network
config = SimulationConfig(network_connectivity=6)
```

Higher connectivity generally:
- Speeds up recovery spread
- Reduces variability between households
- Makes community more resilient

## Comparing Networks

Run sensitivity analysis across network types:

```python
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.config import SimulationConfig

network_types = ['barabasi_albert', 'watts_strogatz', 'erdos_renyi', 'random_geometric']

results = {}
for net_type in network_types:
    config = SimulationConfig(
        num_households=100,
        steps=25,
        network_type=net_type,
        network_connectivity=4
    )

    mc = run_monte_carlo(config, n_runs=50, parallel=True)
    results[net_type] = mc.get_summary()

    print(f"{net_type}:")
    print(f"  Mean: {results[net_type]['final_recovery']['mean']:.3f}")
    print(f"  Std:  {results[net_type]['final_recovery']['std']:.3f}")
```

## Visualizing Networks

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.visualization import plot_network

# Create and visualize different networks
for net_type in ['barabasi_albert', 'watts_strogatz']:
    config = SimulationConfig(
        num_households=30,
        network_type=net_type,
        steps=1
    )

    engine = SimulationEngine(config)
    result = engine.run()

    plot_network(
        result.final_network,
        title=f"{net_type.replace('_', ' ').title()} Network",
        save_path=f"network_{net_type}.png"
    )
```

## Network Statistics

```python
stats = result.final_network.get_statistics()

print(f"Network statistics:")
print(f"  Households: {stats['network']['num_households']}")
print(f"  Edges: {stats['network']['num_edges']}")
print(f"  Avg degree: {stats['network']['avg_degree']:.2f}")
```

## Infrastructure/Business Connections

Beyond household-household connections, the simulation includes:
- Household-Infrastructure connections
- Household-Business connections

These are controlled by `connection_probability`:

```yaml
network:
  connection_probability: 0.6  # 60% chance of connection
```

Higher probability = more households connected to services

## Example: Network Comparison Study

```python
from household_recovery import SimulationEngine, SimulationConfig
from household_recovery.monte_carlo import run_monte_carlo
from household_recovery.visualization import plot_monte_carlo_trajectory
import matplotlib.pyplot as plt

networks = {
    'barabasi_albert': 'Scale-Free',
    'watts_strogatz': 'Small-World',
    'erdos_renyi': 'Random',
    'random_geometric': 'Spatial'
}

fig, ax = plt.subplots(figsize=(10, 6))

for net_type, label in networks.items():
    config = SimulationConfig(
        num_households=100,
        steps=30,
        network_type=net_type,
        network_connectivity=4
    )

    results = run_monte_carlo(config, n_runs=50, parallel=True)

    # Plot mean trajectory
    ax.plot(results.mean_trajectory, label=label)
    ax.fill_between(
        range(len(results.mean_trajectory)),
        results.ci_lower,
        results.ci_upper,
        alpha=0.1
    )

ax.set_xlabel('Time Step')
ax.set_ylabel('Average Recovery')
ax.set_title('Recovery by Network Type')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.savefig('network_comparison.png', dpi=150, bbox_inches='tight')
```

## Key Insights

### Scale-Free (Barabasi-Albert)
- Hub households are critical
- Targeting interventions at hubs is effective
- Most vulnerable to targeted disruption of hubs

### Small-World (Watts-Strogatz)
- Recovery spreads rapidly once started
- Local clusters help each other
- Good for studying community-based recovery

### Random (Erdos-Renyi)
- Most homogeneous outcomes
- Individual factors dominate
- Good control condition

### Spatial (Random Geometric)
- Geographic clustering matters
- Infrastructure location affects recovery
- Most realistic for physical disasters

## Next Steps

- [Monte Carlo](monte-carlo.md) - Statistical analysis
- [Visualization](visualization.md) - Plotting networks
- [Custom Parameters](custom-parameters.md) - Fine-tuning
