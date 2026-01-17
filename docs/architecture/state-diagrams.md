# State Diagrams and Decision Flows

This document provides visual diagrams of the RecovUS decision model's state machine and key decision flows.

## RecovUS State Machine

The RecovUS model uses a probabilistic state machine to model household recovery decisions.

### State Transition Diagram

```
                                    ┌─────────────────────────────────────────┐
                                    │                                         │
                                    ▼                                         │
┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│                 │    r1     │                 │    r2     │                 │
│     WAITING     │ ────────► │    REPAIRING    │ ────────► │    RECOVERED    │
│                 │  (95%)    │                 │  (95%)    │                 │
└────────┬────────┘           └─────────────────┘           └─────────────────┘
         │                            ▲
         │ r0 (35%)                   │
         │ feasible only              │
         └────────────────────────────┘
         │
         │ relocate (5%)
         │ not feasible
         ▼
┌─────────────────┐
│                 │
│    RELOCATED    │
│                 │
└─────────────────┘
```

### State Descriptions

| State | Description | Recovery Level |
|-------|-------------|----------------|
| **WAITING** | Household has not started repairs | 0.0 - 0.3 |
| **REPAIRING** | Actively repairing home | 0.3 - 0.9 |
| **RECOVERED** | Repairs complete, household stable | 0.9 - 1.0 |
| **RELOCATED** | Permanently left the community | N/A |

### Transition Conditions

| Transition | Probability | Conditions |
|------------|-------------|------------|
| WAITING → REPAIRING (r1) | 95% | Financially feasible AND community adequate |
| WAITING → REPAIRING (r0) | 35% | Financially feasible only (community not adequate) |
| REPAIRING → RECOVERED (r2) | 95% | Community adequate, per time step |
| WAITING → RELOCATED | 5% | Not financially feasible, per time step |

## Decision Flow: Should Household Start Repairs?

This flow shows how a household in the WAITING state decides whether to start repairs.

```
                    ┌─────────────────────┐
                    │   Household in      │
                    │   WAITING state     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Is financially     │
                    │  feasible?          │
                    │  (resources >= cost)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │ NO                              │ YES
              ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │  Random check:      │          │  Is community       │
    │  relocate?          │          │  adequate?          │
    │  (5% probability)   │          │  (for perception    │
    └──────────┬──────────┘          │   type)             │
               │                     └──────────┬──────────┘
     ┌─────────┴─────────┐                      │
     │ YES               │ NO      ┌────────────┴────────────┐
     ▼                   ▼         │ NO                      │ YES
┌──────────┐      ┌──────────┐     ▼                         ▼
│ RELOCATED│      │ Stay in  │  ┌─────────────────┐   ┌─────────────────┐
└──────────┘      │ WAITING  │  │  Random check:  │   │  Random check:  │
                  └──────────┘  │  start anyway?  │   │  start repairs? │
                                │  (r0 = 35%)     │   │  (r1 = 95%)     │
                                └────────┬────────┘   └────────┬────────┘
                                         │                     │
                                ┌────────┴────────┐   ┌────────┴────────┐
                                │ YES        NO   │   │ YES        NO   │
                                ▼                 ▼   ▼                 ▼
                          ┌──────────┐     ┌──────────┐          ┌──────────┐
                          │REPAIRING │     │ WAITING  │          │ WAITING  │
                          └──────────┘     └──────────┘          └──────────┘
```

## Financial Feasibility Calculation

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL FEASIBILITY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     TOTAL RESOURCES                           │   │
│  │                                                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────┐ │   │
│  │  │Insurance│ +│ FEMA-HA │ +│SBA Loan │ +│ Liquid  │ +│CDBG │ │   │
│  │  │ Payout  │  │  Grant  │  │ Amount  │  │ Assets  │  │ -DR │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ≥                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                       TOTAL COSTS                             │   │
│  │                                                               │   │
│  │  ┌─────────────────────────┐  ┌────────────────────────────┐ │   │
│  │  │      Repair Cost        │ +│ Temporary Housing Costs    │ │   │
│  │  │ (damage % × home value) │  │ (months × monthly rate)    │ │   │
│  │  └─────────────────────────┘  └────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  If RESOURCES >= COSTS: FEASIBLE = TRUE                             │
│  Otherwise: FEASIBLE = FALSE                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Resource Eligibility Rules

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RESOURCE ELIGIBILITY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INSURANCE PAYOUT                                                    │
│  ├── Has insurance? (60% probability by default)                    │
│  └── If yes: payout = damage_amount × coverage_rate (60-80%)        │
│                                                                      │
│  FEMA HOUSING ASSISTANCE                                             │
│  ├── Gap exists? (repair_cost > insurance_payout)                   │
│  └── If yes: grant = min(gap, $35,500)                              │
│                                                                      │
│  SBA DISASTER LOAN                                                   │
│  ├── Income >= $30,000? (SBA income floor)                          │
│  ├── If yes: eligible for loan                                      │
│  └── Takes loan? (40% uptake rate)                                  │
│      └── If yes: loan = min(repair_cost, $200,000)                  │
│                                                                      │
│  LIQUID ASSETS                                                       │
│  └── Available = net_worth × random(1%, 20%)                        │
│                                                                      │
│  CDBG-DR (Community Development Block Grant)                         │
│  ├── Low-income household? (income < threshold)                     │
│  ├── If yes: eligible for CDBG-DR                                   │
│  └── Receives grant? (30% probability)                              │
│      └── If yes: grant = repair_cost × 50%                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Community Adequacy Evaluation

Each household evaluates community adequacy based on their perception type.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   COMMUNITY ADEQUACY CHECK                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐                                           │
│  │  Household           │                                           │
│  │  perception_type?    │                                           │
│  └──────────┬───────────┘                                           │
│             │                                                        │
│    ┌────────┴────────┬────────────────────┐                         │
│    │                 │                    │                         │
│    ▼                 ▼                    ▼                         │
│  ┌─────────┐    ┌─────────┐         ┌─────────┐                     │
│  │ INFRA-  │    │ SOCIAL  │         │COMMUNITY│                     │
│  │STRUCTURE│    │         │         │         │                     │
│  └────┬────┘    └────┬────┘         └────┬────┘                     │
│       │              │                   │                          │
│       ▼              ▼                   ▼                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │ Check:       │ │ Check:       │ │ Check:       │                 │
│  │ avg_infra_   │ │ avg_neighbor_│ │ avg_business_│                 │
│  │ func >= 0.50 │ │ recovery     │ │ avail >= 0.50│                 │
│  │              │ │ >= 0.40      │ │              │                 │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                 │
│         │                │                │                         │
│         └────────────────┴────────────────┘                         │
│                          │                                          │
│                          ▼                                          │
│                   ┌─────────────┐                                   │
│                   │  ADEQUATE?  │                                   │
│                   │  (TRUE/     │                                   │
│                   │   FALSE)    │                                   │
│                   └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Default Adequacy Thresholds

| Perception Type | Checks | Threshold |
|-----------------|--------|-----------|
| **Infrastructure** | `avg_infra_func` | ≥ 0.50 (50%) |
| **Social** | `avg_neighbor_recovered_binary` | ≥ 0.40 (40%) |
| **Community** | `avg_business_avail` | ≥ 0.50 (50%) |

## Heuristic Modification Points

RAG-extracted heuristics can modify transition probabilities and adequacy thresholds.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HEURISTIC APPLICATION                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For each active heuristic:                                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  IF condition matches context:                                  │ │
│  │                                                                 │ │
│  │    Modify Transition Probabilities:                             │ │
│  │    ├── modify_r0: r0 = r0 × modifier  (e.g., 0.8 = reduce 20%) │ │
│  │    ├── modify_r1: r1 = r1 × modifier  (e.g., 1.2 = increase 20%)│ │
│  │    └── modify_r2: r2 = r2 × modifier                           │ │
│  │                                                                 │ │
│  │    Modify Adequacy Thresholds:                                  │ │
│  │    ├── modify_adq_infr: threshold += modifier  (e.g., -0.1)    │ │
│  │    ├── modify_adq_nbr: threshold += modifier                   │ │
│  │    └── modify_adq_cas: threshold += modifier                   │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Example Heuristic:                                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ IF perception_type == 'social' AND avg_neighbor_recovery > 0.5 │ │
│  │ THEN modify_r1: 1.15, modify_adq_nbr: -0.05                    │ │
│  │                                                                 │ │
│  │ Effect: Social households with recovering neighbors            │ │
│  │         are 15% more likely to start repairs (r1)              │ │
│  │         and need 5% fewer neighbors recovered (lower threshold)│ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Full Simulation Step Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SIMULATION STEP                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. FOR EACH HOUSEHOLD:                                              │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  a. Build context (neighbor recovery, infra, business)    │   │
│     │  b. Evaluate matching heuristics                          │   │
│     │  c. Apply heuristic modifications to probabilities        │   │
│     │  d. Decision model: decide() → (new_recovery, action)     │   │
│     │  e. Update household state and recovery level             │   │
│     │  f. Record decision in history                            │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  2. FOR EACH INFRASTRUCTURE NODE:                                    │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  a. Get connected households                              │   │
│     │  b. Calculate improvement based on household recovery     │   │
│     │  c. Update functionality level                            │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  3. FOR EACH BUSINESS NODE:                                          │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  a. Get connected households                              │   │
│     │  b. Calculate improvement based on household recovery     │   │
│     │  c. Update availability level                             │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
│  4. RECORD STATE:                                                    │
│     ┌───────────────────────────────────────────────────────────┐   │
│     │  a. Record household states                               │   │
│     │  b. Record infrastructure functionality                   │   │
│     │  c. Record business availability                          │   │
│     │  d. Calculate and return average recovery                 │   │
│     └───────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Parameter Precedence Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARAMETER PRECEDENCE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                │
│  │  RAG Extraction │  ← Highest priority (if confidence >= 0.7)    │
│  │  from Papers    │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  Check          │                                                │
│  │  Confidence     │                                                │
│  │  >= 0.7?        │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│    ┌──────┴──────┐                                                  │
│    │ YES         │ NO                                               │
│    ▼             ▼                                                  │
│  ┌──────────┐  ┌─────────────────┐                                  │
│  │Use RAG   │  │  Config File    │  ← Second priority               │
│  │Value     │  │  Value          │                                  │
│  └──────────┘  └────────┬────────┘                                  │
│                         │                                            │
│                         ▼                                            │
│                ┌─────────────────┐                                  │
│                │  Value in       │                                  │
│                │  config file?   │                                  │
│                └────────┬────────┘                                  │
│                         │                                            │
│                  ┌──────┴──────┐                                    │
│                  │ YES         │ NO                                 │
│                  ▼             ▼                                    │
│                ┌──────────┐  ┌─────────────────┐                    │
│                │Use Config│  │  Hardcoded      │  ← Lowest priority │
│                │Value     │  │  Default        │                    │
│                └──────────┘  └─────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Related Documentation

- [RecovUS User Guide](../user-guide/recovus.md) - Configuration and usage
- [Agent Model Architecture](agent-model.md) - Agent design details
- [RAG Architecture](rag-architecture.md) - Parameter extraction pipeline
