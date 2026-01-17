# Data Flow

Detailed data flow diagrams for the simulation.

## Complete Simulation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                 │
│  • CLI arguments                                                  │
│  • Config file (YAML/JSON)                                        │
│  • Python API calls                                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LOADING                          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Simulation   │  │ Threshold    │  │ API          │            │
│  │ Config       │  │ Config       │  │ Config       │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Infra        │  │ Network      │  │ Research     │            │
│  │ Config       │  │ Config       │  │ Config       │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  SIMULATION ENGINE INIT                           │
│                                                                   │
│  SimulationEngine(config, api_config, research_config, ...)      │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│   KNOWLEDGE BASE       │      │   NETWORK CREATION     │
│                        │      │                        │
│  If API keys valid:    │      │  1. Create base graph  │
│    → RAG Pipeline      │      │  2. Generate agents    │
│  Else:                 │      │  3. Connect to infra   │
│    → Fallback          │      │  4. Connect to biz     │
│                        │      │                        │
│  Output: [Heuristics]  │      │  Output: Network       │
└────────────────────────┘      └────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SIMULATION LOOP                               │
│                                                                   │
│  for step in range(1, steps + 1):                                │
│      ┌──────────────────────────────────────────────────────┐    │
│      │  FOR EACH HOUSEHOLD:                                  │    │
│      │                                                       │    │
│      │  1. Build SimulationContext                           │    │
│      │     • avg_neighbor_recovery                           │    │
│      │     • avg_infra_func                                  │    │
│      │     • avg_business_avail                              │    │
│      │     • household attributes                            │    │
│      │                                                       │    │
│      │  2. Evaluate heuristics against context               │    │
│      │     • Check condition (safe_eval)                     │    │
│      │     • Accumulate boosts/extra_recovery                │    │
│      │                                                       │    │
│      │  3. Calculate proposed recovery (utility model only)  │    │
│      │     increment = base_rate * boost + extra             │    │
│      │     proposed = current + increment (capped at 1.0)    │    │
│      │                                                       │    │
│      │  4. Decision model                                    │    │
│      │     • RecovUS: feasibility + adequacy + state machine │    │
│      │     • Utility: accept if utility increases            │    │
│      └──────────────────────────────────────────────────────┘    │
│                                                                   │
│      ┌──────────────────────────────────────────────────────┐    │
│      │  UPDATE INFRASTRUCTURE & BUSINESSES:                  │    │
│      │                                                       │    │
│      │  new_func = old_func + improvement_rate               │    │
│      │           + household_multiplier * avg_hh_recovery    │    │
│      └──────────────────────────────────────────────────────┘    │
│                                                                   │
│      Record state for all agents                                  │
│      Calculate avg_recovery                                       │
│      Append to recovery_history                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SIMULATION RESULT                             │
│                                                                   │
│  • config: SimulationConfig                                       │
│  • recovery_history: [0.0, 0.08, 0.15, ...]                      │
│  • final_network: CommunityNetwork                                │
│  • heuristics_used: [Heuristic, ...]                             │
│  • timing: start_time, end_time                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                     │
│                                                                   │
│  • Console output (recovery per step)                             │
│  • CSV export (trajectories)                                      │
│  • JSON export (full metadata)                                    │
│  • Visualization (plots, reports)                                 │
└──────────────────────────────────────────────────────────────────┘
```

## RAG Pipeline Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
│  • Search query (default or custom)                             │
│  • API keys (SerpAPI, Groq)                                     │
│  • Cache directory                                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                   SCHOLAR RETRIEVER                             │
│                                                                 │
│  1. Check cache                                                 │
│     If valid cache exists: return cached papers                 │
│                                                                 │
│  2. Call SerpAPI                                                │
│     params = {engine: google_scholar, q: query, num: 5}         │
│                                                                 │
│  3. Parse results into Paper objects                            │
│     Paper(title, abstract/text, authors, year, link, cited_by)   │
│                                                                 │
│  4. Cache results                                               │
│                                                                 │
│  Output: [Paper, Paper, ...]                                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  HEURISTIC EXTRACTOR                            │
│                                                                 │
│  1. Format text excerpts into prompt                            │
│     "Extract 4-6 actionable heuristics..."                      │
│     Include: allowed context keys, output format                │
│                                                                 │
│  2. Call Groq LLM                                               │
│     model: llama-3.3-70b-versatile                              │
│     temperature: 0.05 (deterministic)                           │
│                                                                 │
│  3. Parse JSON response                                         │
│     [{"condition": "...", "action": {...}, "source": "..."}]    │
│                                                                 │
│  4. Validate each heuristic                                     │
│     • Check required keys present                               │
│     • Validate condition with safe_eval                         │
│     • Compile condition into callable                           │
│                                                                 │
│  Output: [Heuristic(compiled), ...]                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                 PARAMETER EXTRACTOR (Optional)                  │
│                                                                 │
│  1. Format text excerpts into parameter prompt                  │
│     "Extract numeric parameters: recovery rates, thresholds..." │
│                                                                 │
│  2. Call LLM                                                    │
│                                                                 │
│  3. Parse and validate                                          │
│     • Check bounds (0.01 < recovery_rate < 0.5)                 │
│     • Extract confidence scores                                 │
│                                                                 │
│  Output: ExtractedParameters                                    │
└────────────────────────────────────────────────────────────────┘
```

## Parameter Precedence Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    PARAMETER MERGER                             │
│                                                                 │
│  For each parameter (e.g., base_recovery_rate):                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Check RAG-extracted                                   │  │
│  │     if extracted.base_recovery_rate is not None           │  │
│  │        and confidence >= 0.7:                             │  │
│  │        USE extracted value                                │  │
│  │        SOURCE = "RAG-extracted (confidence=X.XX)"         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │ (else)                                │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. Check config file                                     │  │
│  │     if config.base_recovery_rate != default:              │  │
│  │        USE config value                                   │  │
│  │        SOURCE = "config file"                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │ (else)                                │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. Use default                                           │  │
│  │     USE hardcoded default (0.1)                           │  │
│  │     SOURCE = "default"                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Log all merge decisions for transparency                       │
└────────────────────────────────────────────────────────────────┘
```

## Monte Carlo Flow

```
┌────────────────────────────────────────────────────────────────┐
│                     MONTE CARLO RUNNER                          │
│                                                                 │
│  1. Build heuristics ONCE (shared across all runs)              │
│                                                                 │
│  2. Serialize heuristics and configs for multiprocessing        │
│                                                                 │
│  3. For i in range(n_runs):                                     │
│     ┌──────────────────────────────────────────────────────┐   │
│     │  Create unique seed: base_seed + i                    │   │
│     │  Run SimulationEngine with seed                       │   │
│     │  Collect SimulationResult                             │   │
│     └──────────────────────────────────────────────────────┘   │
│     (parallel or sequential)                                    │
│                                                                 │
│  4. Aggregate results:                                          │
│     • Stack all recovery_history arrays                         │
│     • Compute mean, std, CI per step                            │
│                                                                 │
│  Output: MonteCarloResults                                      │
└────────────────────────────────────────────────────────────────┘
```

## Next Steps

- [Module Relationships](module-relationships.md) - Import dependencies
- [RAG Architecture](rag-architecture.md) - Extraction details
- [Agent Model](agent-model.md) - Decision logic
