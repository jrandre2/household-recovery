# Household Recovery Simulation: Use Cases & Applications

## What Makes This System Different

Most disaster recovery models require researchers to manually translate academic findings into code. This system automates that process: point it at research papers, and it extracts behavioral rules and parameters for simulation.

| Traditional ABM | This System |
|-----------------|-------------|
| Read papers → manually code rules → simulation becomes stale | RAG pipeline extracts rules → simulation evolves with literature |
| Parameters are researcher opinions | Parameters traced to specific studies with confidence scores |
| One model, one disaster type | 17 pre-built configurations from Katrina, Andrew, Ike, Camp Fire, etc. |
| Trust the code | Safe evaluation layer validates LLM-generated conditions |

---

## The RAG-to-Simulation Pipeline

### How It Works

1. **Input**: Point the system at Google Scholar queries or local PDF folders
2. **Extraction**: LLM (Groq) reads papers and generates behavioral heuristics as IF-THEN rules
3. **Validation**: AST-based safe evaluation checks that rules are executable and secure
4. **Integration**: 3-tier parameter precedence merges RAG-extracted values with config defaults

### Example Extraction

From a paper on Hurricane Katrina recovery:

> "Households with strong neighborhood ties were 2.3x more likely to return within the first year"

The system extracts:
```yaml
condition: "ctx['avg_neighbor_recovery'] > 0.5 and ctx['social_network_strength'] > 0.7"
action: {boost: 2.3}
source: "Aldrich 2012, Table 4"
confidence: 0.85
```

### Why This Matters

**Research findings become immediately testable.** When a new disaster recovery paper is published, you can feed it to the system and see how its findings would alter simulation outcomes—without writing code.

**Parameters are auditable.** Every value traces back to a source. When stakeholders ask "why does the model assume X?", you can point to the paper, page, and confidence level.

**The model updates with new research.** As you add more papers, the system's behavioral repertoire expands. A 2024 paper on wildfire recovery can immediately inform simulations alongside 2005 Katrina research.

---

## The RecovUS Decision Model

This implements a specific theory of household recovery with empirically-derived components:

### Perception Types (ASNA Framework)

Households perceive recovery readiness through different lenses:

| Type | Prevalence | What They Monitor |
|------|------------|-------------------|
| **Infrastructure** | 65% | Power, water, roads, services |
| **Social** | 31% | Neighbor recovery, community return |
| **Community** | 4% | Business reopening, institutional recovery |

A household won't begin repairs until their perception-specific threshold is met—even if other conditions are favorable.

### Financial Feasibility Model

Recovery requires assembling funds from 5 distinct sources, each with different timing and eligibility:

1. **Insurance** — Fast but coverage varies
2. **FEMA Individual Assistance** — Quick, small amounts
3. **SBA Disaster Loans** — Larger but requires creditworthiness
4. **Liquid Assets** — Immediate but depletes savings
5. **CDBG-DR** — Large grants but arrives 18-24 months post-disaster

The model tracks which resources each household can access and when, creating realistic financial bottlenecks.

### State Machine Transitions

Households move through states with calibrated probabilities:

```
WAITING ──(r0=0.35)──► REPAIRING ──(r1=0.95)──► RECOVERED
    │                      │
    └──────────────────────┴──(r2=0.95)──► RELOCATED
```

These transition probabilities are derived from longitudinal recovery studies.

---

## Practical Applications

### 1. Testing How Research Findings Generalize

**Scenario**: A city wants to apply lessons from Hurricane Harvey recovery to their flood planning.

**Approach**:
- Load Harvey research papers into the RAG pipeline
- Run simulations with your city's demographics and infrastructure
- See which Harvey-derived heuristics improve outcomes in your context vs. which don't transfer

**Output**: Evidence-based assessment of which recovery strategies are context-dependent vs. generalizable.

---

### 2. Identifying Financial Bottleneck Timing

**Scenario**: A state is designing a new disaster recovery fund. When should it disburse?

**Use the financial feasibility model to simulate**:
- Insurance pays out Month 1-3
- FEMA arrives Month 1-2
- SBA loans approved Month 3-6
- CDBG-DR arrives Month 18-24

**Find the gap**: Simulations reveal households stall in Month 4-8 when insurance is exhausted and CDBG-DR hasn't arrived. A state bridge fund targeting Month 4 would have 3x the impact of the same fund at Month 12.

---

### 3. Understanding Perception-Type Disparities

**Scenario**: Why do some neighborhoods recover quickly while similar ones stall?

**Use perception type analysis**:
- Infrastructure-focused households (65%) wait for utilities—infrastructure repair accelerates them
- Social-focused households (31%) wait for neighbors—these neighborhoods have "chicken and egg" problems
- Community-focused households (4%) wait for businesses—commercial recovery unlocks them

**Intervention design**: A neighborhood with high social-focus concentration needs different interventions (return incentives, community events) than one with high infrastructure-focus (generator programs, utility prioritization).

---

### 4. Validating Heuristics Across Disaster Types

**Scenario**: Do wildfire recovery patterns differ from hurricane recovery?

**Load research from both disaster types**:
- Katrina, Harvey, Andrew papers → hurricane heuristics
- Camp Fire, Tubbs Fire papers → wildfire heuristics

**Compare extracted rules**: The system might find that social network effects are stronger in hurricane recovery (evacuation is temporary) while infrastructure effects dominate wildfire recovery (total destruction requires rebuilding from zero).

**Output**: Disaster-type-specific parameter sets with documented sources.

---

### 5. Stress-Testing Policy Robustness

**Scenario**: A proposed policy performs well in simulations. But is it robust?

**Use Monte Carlo with parameter uncertainty**:
- Run 500 simulations varying RAG-extracted parameters within their confidence intervals
- A policy that works when `neighbor_influence = 0.4` but fails at `0.3` is fragile
- A policy that works across the plausible parameter range is robust

**Output**: Confidence intervals on policy effectiveness, not point estimates.

---

### 6. Rapid Literature Review Operationalization

**Scenario**: A research team completed a systematic review of 50 disaster recovery papers. Now what?

**Approach**:
- Feed all 50 PDFs to the RAG pipeline
- Extract and deduplicate heuristics across papers
- Identify where papers agree (high-confidence parameters) vs. contradict (research gaps)
- Generate a simulation that embodies the entire literature

**Output**: A computational model representing the literature review—testable, comparable, and updatable.

---

## Key Capabilities

1. **Living models** that evolve with published research
2. **Traceable assumptions** where every parameter links to evidence
3. **Cross-disaster learning** by systematically comparing extracted rules
4. **Safe LLM integration** where generated code is validated before execution
5. **Perception-aware policy design** targeting households based on how they evaluate recovery readiness
6. **Financial timing optimization** using the 5-source feasibility model

---

## Limitations

- **Calibration required**: RAG-extracted parameters are starting points. Validate against local historical data before policy use.
- **Confidence thresholds matter**: Parameters below 0.7 confidence fall back to defaults. Low-confidence extractions flag research gaps, not definitive values.
- **Scenario exploration, not prediction**: This explores "what if" scenarios, not "what will happen."
- **Input quality matters**: RAG extraction quality depends on paper quality. Preprints and gray literature may yield unreliable heuristics.

---

## Summary

This system automates the extraction of behavioral rules from disaster recovery research and runs them as simulations. Researchers can load academic papers, extract empirically-grounded parameters, and test how those findings play out across different community configurations.

The three core components:
- **RAG pipeline** extracts IF-THEN heuristics from papers with source attribution and confidence scores
- **RecovUS model** implements perception-based decision-making and multi-source financial feasibility
- **Safe evaluation** validates LLM-generated conditions before execution

The result: a simulation framework where parameters trace to published research, models update as new papers are added, and practitioners can test whether findings from one disaster context transfer to another.
