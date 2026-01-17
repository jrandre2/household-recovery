"""
Heuristic extraction and management.

This module handles:
1. Fetching academic papers from Google Scholar (via SerpApi)
2. Extracting behavioral heuristics using LLM (Groq)
3. Validating and compiling heuristics safely

Educational Note:
-----------------
This implements a RAG (Retrieval-Augmented Generation) pattern:
1. RETRIEVE: Fetch relevant academic papers from Google Scholar
2. AUGMENT: Use the paper text as context for the LLM
3. GENERATE: LLM extracts actionable heuristics from the research

This allows the simulation to be grounded in actual research findings
rather than arbitrary assumptions.
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Heuristic:
    """
    A behavioral rule extracted from research.

    Heuristics have the form:
        IF <condition> THEN <action>

    Example:
        IF neighbors mostly recovered THEN boost recovery by 50%
    """
    condition_str: str
    action: dict[str, float]
    source: str
    _evaluator: Callable[[dict], bool] | None = field(default=None, repr=False)

    def evaluate(self, ctx: dict[str, Any]) -> bool:
        """Evaluate the condition against a context dictionary."""
        if self._evaluator is None:
            raise RuntimeError("Heuristic not compiled - call compile() first")
        return self._evaluator(ctx)

    def compile(self) -> Heuristic:
        """Compile the condition string into an evaluator function."""
        from .safe_eval import compile_condition
        self._evaluator = compile_condition(self.condition_str)
        return self


@dataclass
class Paper:
    """Represents an academic paper retrieved from Google Scholar."""
    title: str
    abstract: str
    authors: str
    year: str
    link: str
    cited_by: int = 0


class ScholarRetriever:
    """Fetches papers from Google Scholar via SerpApi."""

    def __init__(self, api_key: str, cache_dir: Path | None = None, cache_expiry_hours: int = 24):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_expiry = timedelta(hours=cache_expiry_hours)

    def search(self, query: str, num_results: int = 5) -> list[Paper]:
        """
        Search Google Scholar for papers matching the query.

        Args:
            query: Search query string
            num_results: Maximum number of papers to return

        Returns:
            List of Paper objects
        """
        # Check cache first
        if self.cache_dir:
            cached = self._load_from_cache(query)
            if cached:
                logger.info(f"Loaded {len(cached)} papers from cache")
                return cached

        # Fetch from API
        papers = self._fetch_from_api(query, num_results)

        # Cache results
        if self.cache_dir and papers:
            self._save_to_cache(query, papers)

        return papers

    def _fetch_from_api(self, query: str, num_results: int) -> list[Paper]:
        """Fetch papers from SerpApi."""
        if not self.api_key:
            logger.warning("No SerpApi key provided")
            return []

        try:
            from serpapi import GoogleSearch
        except ImportError:
            logger.error("serpapi package not installed. Run: pip install google-search-results")
            return []

        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])

            papers = []
            for r in results:
                paper = Paper(
                    title=r.get("title", "No title"),
                    abstract=r.get("snippet", "No abstract"),
                    authors=r.get("publication_info", {}).get("authors", "Unknown authors"),
                    year=r.get("publication_info", {}).get("year", "Unknown year"),
                    link=r.get("link", "No link"),
                    cited_by=r.get("inline_links", {}).get("cited_by", {}).get("total", 0)
                )
                papers.append(paper)

            logger.info(f"Fetched {len(papers)} papers from Google Scholar")
            return papers

        except Exception as e:
            logger.error(f"SerpApi error: {e}")
            return []

    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _load_from_cache(self, query: str) -> list[Paper] | None:
        """Load papers from cache if not expired."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.cache_expiry:
                logger.info("Cache expired")
                return None

            return [Paper(**p) for p in data['papers']]

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _save_to_cache(self, query: str, papers: list[Paper]) -> None:
        """Save papers to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{self._cache_key(query)}.json"
        data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'papers': [
                {
                    'title': p.title,
                    'abstract': p.abstract,
                    'authors': p.authors,
                    'year': p.year,
                    'link': p.link,
                    'cited_by': p.cited_by
                }
                for p in papers
            ]
        }

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached {len(papers)} papers")
        except IOError as e:
            logger.warning(f"Cache write error: {e}")


class HeuristicExtractor:
    """Extracts behavioral heuristics from paper text using LLM."""

    PROMPT_TEMPLATE = """You are a strict JSON generator. Return ONLY a valid JSON array – nothing else.

Task: Extract 4–6 actionable heuristics for utility-based household agents in disaster recovery simulations.

Rules:
- condition: valid Python expression using ONLY these ctx keys:
  'avg_neighbor_recovery', 'avg_infra_func', 'avg_business_avail', 'num_neighbors',
  'resilience', 'resilience_category', 'household_income', 'income_level'
- DO NOT use any other keys
- action: dict like {{"boost": 1.5}} or {{"extra_recovery": 0.1}}
- source: very short string

Output format – start directly with array:

[
  {{"condition": "ctx['avg_neighbor_recovery'] > 0.5", "action": {{"boost": 1.5}}, "source": "social influence"}},
  ...
]

Text excerpts:
{texts}"""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.05,
        max_tokens: int = 1200,
        chunk_size_chars: int = 8000,
        chunk_overlap_chars: int = 800,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def extract(self, papers: list[Paper]) -> list[Heuristic]:
        """
        Extract heuristics from paper text.

        Args:
            papers: List of papers with text content

        Returns:
            List of compiled Heuristic objects
        """
        if not papers:
            logger.warning("No papers provided for heuristic extraction")
            return []

        if not self.api_key:
            logger.warning("No Groq API key - cannot extract heuristics")
            return []

        all_heuristics: list[Heuristic] = []
        for paper in papers:
            text = paper.abstract or ""
            if not text.strip():
                continue

            chunks = _chunk_text(text, self.chunk_size_chars, self.chunk_overlap_chars)
            if len(chunks) > 1:
                logger.info(
                    f"Split '{paper.title[:60]}' into {len(chunks)} chunks for extraction"
                )

            for chunk in chunks:
                formatted = f"Title: {paper.title}\nText: {chunk}"
                prompt = self.PROMPT_TEMPLATE.format(texts=formatted)

                raw_heuristics = self._call_llm(prompt)
                compiled = self._compile_heuristics(raw_heuristics, log=False)
                all_heuristics.extend(compiled)

        if not all_heuristics:
            return []

        deduped = _dedupe_heuristics(all_heuristics)
        logger.info(f"Compiled {len(deduped)} unique heuristics")
        return deduped

    def _call_llm(self, prompt: str) -> list[dict]:
        """Call Groq LLM and parse response."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            logger.error("langchain-groq package not installed. Run: pip install langchain-groq")
            return []

        import os
        os.environ["GROQ_API_KEY"] = self.api_key

        llm = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(method="json_mode")
            response = structured_llm.invoke(prompt)

            if isinstance(response, list):
                logger.info(f"Structured mode: {len(response)} heuristics")
                return response
            elif isinstance(response, dict):
                heuristics = response.get("heuristics", []) or list(response.values())[0] if response else []
                logger.info(f"Structured mode (wrapped): {len(heuristics)} heuristics")
                return heuristics

        except Exception as e:
            logger.warning(f"Structured mode failed: {e}")

        # Fallback to raw call
        try:
            raw_response = llm.invoke(prompt).content.strip()
            return self._parse_raw_response(raw_response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []

    def _parse_raw_response(self, raw: str) -> list[dict]:
        """Parse raw LLM response with aggressive cleaning."""
        cleaned = raw

        # Remove markdown fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        # Find JSON array
        start_idx = cleaned.find("[")
        if start_idx != -1:
            cleaned = cleaned[start_idx:]

        end_idx = cleaned.rfind("]")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                logger.info(f"Parsed {len(parsed)} heuristics from raw response")
                return parsed
            elif isinstance(parsed, dict) and "heuristics" in parsed:
                return parsed["heuristics"]
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")

        return []

    def _compile_heuristics(self, raw_heuristics: list[dict], log: bool = True) -> list[Heuristic]:
        """Validate and compile raw heuristics."""
        from .safe_eval import validate_condition

        compiled = []
        for i, h in enumerate(raw_heuristics, 1):
            if not isinstance(h, dict):
                logger.warning(f"Heuristic {i}: not a dict, skipping")
                continue

            required = {'condition', 'action', 'source'}
            if not required.issubset(h.keys()):
                logger.warning(f"Heuristic {i}: missing keys {required - set(h.keys())}")
                continue

            # Validate condition
            is_valid, error = validate_condition(h['condition'])
            if not is_valid:
                logger.warning(f"Heuristic {i}: invalid condition - {error}")
                continue

            try:
                heuristic = Heuristic(
                    condition_str=h['condition'],
                    action=h['action'],
                    source=h.get('source', 'Generated')
                ).compile()
                compiled.append(heuristic)
            except Exception as e:
                logger.warning(f"Heuristic {i}: compilation failed - {e}")

        if log:
            logger.info(f"Compiled {len(compiled)}/{len(raw_heuristics)} heuristics")
        return compiled


class RecovUSHeuristicExtractor(HeuristicExtractor):
    """Extracts RecovUS-style heuristics that adjust transition probabilities."""

    PROMPT_TEMPLATE = """You are a strict JSON generator. Return ONLY a valid JSON array - nothing else.

Task: Extract 4-6 actionable heuristics for RecovUS household recovery decisions.

Rules:
- condition: valid Python expression using ONLY these ctx keys:
  'avg_neighbor_recovery', 'avg_infra_func', 'avg_business_avail', 'num_neighbors',
  'resilience', 'resilience_category', 'household_income', 'income_level',
  'perception_type', 'damage_severity', 'recovery_state', 'repair_cost',
  'is_habitable', 'available_resources', 'is_feasible', 'time_step',
  'months_since_disaster', 'avg_neighbor_recovered_binary'
- DO NOT use any other keys
- action: dict using one or more of:
  - 'modify_r0', 'modify_r1', 'modify_r2' (multipliers like 0.8, 1.1)
  - 'modify_adq_infr', 'modify_adq_nbr', 'modify_adq_cas' (additive deltas like -0.05, 0.10)
- source: very short string
- categorical fields must be compared to string literals using == or !=
  (perception_type, damage_severity, recovery_state, income_level, resilience_category)

Output format - start directly with array:

[
  {{"condition": "ctx['perception_type'] == 'social' and ctx['avg_neighbor_recovery'] > 0.6",
   "action": {{"modify_r1": 1.15, "modify_adq_nbr": -0.05}},
   "source": "social capital"}},
  ...
]

Text excerpts:
{texts}"""


# Validation bounds for extracted parameters
PARAMETER_VALIDATION_RULES = {
    'base_recovery_rate': {'min': 0.01, 'max': 0.5, 'default': 0.1},
    'income_threshold_low': {'min': 10000, 'max': 100000, 'default': 45000},
    'income_threshold_high': {'min': 50000, 'max': 500000, 'default': 120000},
    'resilience_threshold_low': {'min': 0.1, 'max': 0.5, 'default': 0.35},
    'resilience_threshold_high': {'min': 0.5, 'max': 0.95, 'default': 0.70},
    'utility_weight_self': {'min': 0.5, 'max': 2.0, 'default': 1.0},
    'utility_weight_neighbor': {'min': 0.0, 'max': 1.0, 'default': 0.3},
    'utility_weight_infrastructure': {'min': 0.0, 'max': 1.0, 'default': 0.2},
    'utility_weight_business': {'min': 0.0, 'max': 1.0, 'default': 0.2},
}

# RecovUS-specific validation rules
RECOVUS_VALIDATION_RULES = {
    # Perception distribution (should sum to 1.0)
    'perception_infrastructure': {'min': 0.0, 'max': 1.0, 'default': 0.65},
    'perception_social': {'min': 0.0, 'max': 1.0, 'default': 0.31},
    'perception_community': {'min': 0.0, 'max': 1.0, 'default': 0.04},
    # Adequacy thresholds
    'adequacy_infrastructure': {'min': 0.0, 'max': 1.0, 'default': 0.50},
    'adequacy_neighbor': {'min': 0.0, 'max': 1.0, 'default': 0.40},
    'adequacy_community_assets': {'min': 0.0, 'max': 1.0, 'default': 0.50},
    # Transition probabilities
    'transition_r0': {'min': 0.0, 'max': 1.0, 'default': 0.35},
    'transition_r1': {'min': 0.0, 'max': 1.0, 'default': 0.95},
    'transition_r2': {'min': 0.0, 'max': 1.0, 'default': 0.95},
    'transition_relocate': {'min': 0.0, 'max': 1.0, 'default': 0.05},
}


@dataclass
class ExtractedParameters:
    """
    Numeric parameters extracted from research papers via RAG.

    These parameters can override config file defaults when extracted
    with sufficient confidence from academic literature.
    """
    # Base recovery rate (per simulation step)
    base_recovery_rate: float | None = None
    base_recovery_rate_confidence: float | None = None
    base_recovery_rate_source: str | None = None

    # Income thresholds for classification
    income_threshold_low: float | None = None
    income_threshold_high: float | None = None
    income_thresholds_source: str | None = None
    income_thresholds_confidence: float | None = None

    # Resilience thresholds
    resilience_threshold_low: float | None = None
    resilience_threshold_high: float | None = None
    resilience_thresholds_source: str | None = None
    resilience_thresholds_confidence: float | None = None

    # Utility weights
    utility_weight_self: float | None = None
    utility_weight_neighbor: float | None = None
    utility_weight_infrastructure: float | None = None
    utility_weight_business: float | None = None
    utility_weights_source: str | None = None
    utility_weights_confidence: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        from dataclasses import asdict
        return {k: v for k, v in asdict(self).items() if v is not None}

    def has_any_parameters(self) -> bool:
        """Check if any parameters were extracted."""
        return any([
            self.base_recovery_rate is not None,
            self.income_threshold_low is not None,
            self.resilience_threshold_low is not None,
            self.utility_weight_self is not None,
        ])


@dataclass
class RecovUSExtractedParameters:
    """
    RecovUS-specific parameters extracted from research papers via RAG.

    These parameters configure the RecovUS decision model:
    - Perception distribution (ASNA Index)
    - Community adequacy thresholds
    - State transition probabilities
    """
    # Perception distribution (ASNA Index)
    perception_infrastructure: float | None = None
    perception_social: float | None = None
    perception_community: float | None = None
    perception_confidence: float | None = None
    perception_source: str | None = None

    # Community adequacy thresholds
    adequacy_infrastructure: float | None = None  # adq_infr
    adequacy_neighbor: float | None = None  # adq_nbr
    adequacy_community_assets: float | None = None  # adq_cas
    adequacy_confidence: float | None = None
    adequacy_source: str | None = None

    # State transition probabilities
    transition_r0: float | None = None  # Repair when only feasible
    transition_r1: float | None = None  # Repair when feasible AND adequate
    transition_r2: float | None = None  # Completion probability
    transition_confidence: float | None = None
    transition_source: str | None = None

    # Financial parameters
    insurance_penetration_rate: float | None = None
    fema_ha_average: float | None = None
    sba_uptake_rate: float | None = None
    financial_confidence: float | None = None
    financial_source: str | None = None
    # Disaster context
    disaster_type: str | None = None
    disaster_event: str | None = None
    # Transition relocation probability
    transition_relocate: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        from dataclasses import asdict
        return {k: v for k, v in asdict(self).items() if v is not None}

    def has_any_parameters(self) -> bool:
        """Check if any parameters were extracted."""
        return any([
            self.perception_infrastructure is not None,
            self.adequacy_infrastructure is not None,
            self.transition_r0 is not None,
            self.insurance_penetration_rate is not None,
            self.transition_relocate is not None,
        ])

    def apply_to_config(self, recovus_config: 'RecovUSConfig', confidence_threshold: float = 0.7) -> 'RecovUSConfig':
        """
        Apply extracted parameters to a RecovUSConfig, respecting confidence threshold.

        Args:
            recovus_config: Base configuration to modify
            confidence_threshold: Minimum confidence to apply extracted values

        Returns:
            Modified RecovUSConfig
        """
        from .config import RecovUSConfig
        from dataclasses import asdict

        config_dict = asdict(recovus_config)

        # Apply perception distribution if confident
        if self.perception_confidence and self.perception_confidence >= confidence_threshold:
            if self.perception_infrastructure is not None:
                config_dict['perception_infrastructure'] = self.perception_infrastructure
            if self.perception_social is not None:
                config_dict['perception_social'] = self.perception_social
            if self.perception_community is not None:
                config_dict['perception_community'] = self.perception_community

        # Apply adequacy thresholds if confident
        if self.adequacy_confidence and self.adequacy_confidence >= confidence_threshold:
            if self.adequacy_infrastructure is not None:
                config_dict['adequacy_infrastructure'] = self.adequacy_infrastructure
            if self.adequacy_neighbor is not None:
                config_dict['adequacy_neighbor'] = self.adequacy_neighbor
            if self.adequacy_community_assets is not None:
                config_dict['adequacy_community_assets'] = self.adequacy_community_assets

        # Apply transition probabilities if confident
        if self.transition_confidence and self.transition_confidence >= confidence_threshold:
            if self.transition_r0 is not None:
                config_dict['transition_r0'] = self.transition_r0
            if self.transition_r1 is not None:
                config_dict['transition_r1'] = self.transition_r1
            if self.transition_r2 is not None:
                config_dict['transition_r2'] = self.transition_r2
            if self.transition_relocate is not None:
                config_dict['transition_relocate'] = self.transition_relocate

        # Apply financial parameters if confident
        if self.financial_confidence and self.financial_confidence >= confidence_threshold:
            if self.insurance_penetration_rate is not None:
                config_dict['insurance_penetration_rate'] = self.insurance_penetration_rate
            if self.sba_uptake_rate is not None:
                config_dict['sba_uptake_rate'] = self.sba_uptake_rate

        return RecovUSConfig(**config_dict)


class ParameterExtractor:
    """
    Extracts numeric simulation parameters from paper text using LLM.

    This is separate from heuristic extraction to avoid overloading a single prompt.
    The extracted parameters can be used to configure simulations based on
    empirical findings from disaster recovery literature.
    """

    PROMPT_TEMPLATE = """You are a research data extractor. Analyze these academic paper texts about disaster recovery and extract NUMERIC PARAMETERS mentioned in the research.

Return ONLY a valid JSON object with these fields (use null if not found):

{{
  "base_recovery_rate": {{
    "value": <float 0.0-1.0, monthly recovery rate or per-step increment>,
    "unit": "<monthly|yearly|per_step>",
    "source_quote": "<exact quote from abstract supporting this>",
    "confidence": <0.0-1.0, how confident you are in this extraction>
  }},
  "recovery_timeline_months": {{
    "value": <integer, typical months to full recovery>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }},
  "income_thresholds": {{
    "low_threshold": <float, annual income below which is 'low income' in USD>,
    "high_threshold": <float, annual income above which is 'high income' in USD>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }},
  "resilience_thresholds": {{
    "low_threshold": <float 0.0-1.0, below this is low resilience>,
    "high_threshold": <float 0.0-1.0, above this is high resilience>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }},
  "social_influence_weight": {{
    "value": <float, relative importance of neighbor recovery>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }},
  "infrastructure_weight": {{
    "value": <float, relative importance of infrastructure>,
    "source_quote": "<exact quote>",
    "confidence": <0.0-1.0>
  }}
}}

IMPORTANT EXTRACTION GUIDELINES:
1. Only extract values EXPLICITLY mentioned or clearly implied in the text
2. Convert percentages to decimals (50% -> 0.5)
3. If a paper mentions "80% recovered within 24 months", compute monthly rate as 0.8/24 ≈ 0.033
4. Include the EXACT quote that supports each extraction
5. Set confidence based on how explicit the value is:
   - Direct quote with number = 0.9+
   - Inferred from context = 0.5-0.7
   - Very uncertain = 0.3-0.5
6. Use null for fields where no relevant information is found

Document text to analyze:
{texts}"""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.05,
        max_tokens: int = 2000,
        chunk_size_chars: int = 8000,
        chunk_overlap_chars: int = 800,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def extract(self, papers: list[Paper]) -> ExtractedParameters:
        """
        Extract numeric parameters from paper text.

        Args:
            papers: List of papers with text content

        Returns:
            ExtractedParameters with any values found
        """
        if not papers:
            logger.warning("No papers provided for parameter extraction")
            return ExtractedParameters()

        if not self.api_key:
            logger.warning("No Groq API key - cannot extract parameters")
            return ExtractedParameters()

        merged = ExtractedParameters()

        for paper in papers:
            text = paper.abstract or ""
            if not text.strip():
                continue

            chunks = _chunk_text(text, self.chunk_size_chars, self.chunk_overlap_chars)
            if len(chunks) > 1:
                logger.info(
                    f"Split '{paper.title[:60]}' into {len(chunks)} chunks for parameter extraction"
                )

            for chunk in chunks:
                formatted = f"Title: {paper.title}\nText: {chunk}"
                prompt = self.PROMPT_TEMPLATE.format(texts=formatted)

                raw_params = self._call_llm(prompt)
                candidate = self._validate_and_convert(raw_params, log=False)
                merged = self._merge_params(merged, candidate)

        if merged.has_any_parameters():
            logger.info(f"Extracted parameters: {merged.to_dict()}")
        else:
            logger.info("No numeric parameters extracted from papers")

        return merged

    def _call_llm(self, prompt: str) -> dict:
        """Call Groq LLM and parse response."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            logger.error("langchain-groq package not installed")
            return {}

        import os
        os.environ["GROQ_API_KEY"] = self.api_key

        llm = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            # Try structured output
            structured_llm = llm.with_structured_output(method="json_mode")
            response = structured_llm.invoke(prompt)

            if isinstance(response, dict):
                return response
        except Exception as e:
            logger.warning(f"Structured mode failed for parameters: {e}")

        # Fallback to raw parsing
        try:
            raw_response = llm.invoke(prompt).content.strip()
            return self._parse_raw_response(raw_response)
        except Exception as e:
            logger.error(f"Parameter extraction LLM call failed: {e}")
            return {}

    def _parse_raw_response(self, raw: str) -> dict:
        """Parse raw LLM response."""
        cleaned = raw

        # Remove markdown fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        # Find JSON object
        start_idx = cleaned.find("{")
        if start_idx != -1:
            cleaned = cleaned[start_idx:]

        end_idx = cleaned.rfind("}")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Parameter JSON parse error: {e}")
            return {}

    def _validate_and_convert(self, raw: dict, log: bool = True) -> ExtractedParameters:
        """Validate extracted values and convert to ExtractedParameters."""
        params = ExtractedParameters()

        # Base recovery rate
        if raw.get('base_recovery_rate'):
            brr = raw['base_recovery_rate']
            value = brr.get('value')
            confidence = brr.get('confidence', 0.5)

            # Convert from timeline if not directly available
            if value is None and raw.get('recovery_timeline_months'):
                timeline = raw['recovery_timeline_months'].get('value')
                if timeline and timeline > 0:
                    # Assume ~80% recovery over the timeline
                    value = 0.8 / timeline
                    confidence = min(confidence, 0.6)  # Lower confidence for derived value

            if value is not None:
                value = self._validate_param('base_recovery_rate', value)
                if value is not None:
                    params.base_recovery_rate = value
                    params.base_recovery_rate_confidence = confidence
                    params.base_recovery_rate_source = brr.get('source_quote', 'Extracted from text')

        # Income thresholds
        if raw.get('income_thresholds'):
            inc = raw['income_thresholds']
            low = inc.get('low_threshold')
            high = inc.get('high_threshold')
            confidence = inc.get('confidence', 0.5)

            if low is not None and high is not None:
                low = self._validate_param('income_threshold_low', low)
                high = self._validate_param('income_threshold_high', high)
                if low is not None and high is not None and low < high:
                    params.income_threshold_low = low
                    params.income_threshold_high = high
                    params.income_thresholds_confidence = confidence
                    params.income_thresholds_source = inc.get('source_quote', 'Extracted from text')

        # Resilience thresholds
        if raw.get('resilience_thresholds'):
            res = raw['resilience_thresholds']
            low = res.get('low_threshold')
            high = res.get('high_threshold')
            confidence = res.get('confidence', 0.5)

            if low is not None and high is not None:
                low = self._validate_param('resilience_threshold_low', low)
                high = self._validate_param('resilience_threshold_high', high)
                if low is not None and high is not None and low < high:
                    params.resilience_threshold_low = low
                    params.resilience_threshold_high = high
                    params.resilience_thresholds_confidence = confidence
                    params.resilience_thresholds_source = res.get('source_quote', 'Extracted from text')

        # Utility weights
        has_weights = False
        if raw.get('social_influence_weight'):
            siw = raw['social_influence_weight']
            value = self._validate_param('utility_weight_neighbor', siw.get('value'))
            if value is not None:
                params.utility_weight_neighbor = value
                params.utility_weights_confidence = siw.get('confidence', 0.5)
                params.utility_weights_source = siw.get('source_quote', 'Extracted from text')
                has_weights = True

        if raw.get('infrastructure_weight'):
            iw = raw['infrastructure_weight']
            value = self._validate_param('utility_weight_infrastructure', iw.get('value'))
            if value is not None:
                params.utility_weight_infrastructure = value
                if not has_weights:
                    params.utility_weights_confidence = iw.get('confidence', 0.5)
                    params.utility_weights_source = iw.get('source_quote', 'Extracted from text')

        if log:
            if params.has_any_parameters():
                logger.info(f"Extracted parameters: {params.to_dict()}")
            else:
                logger.info("No numeric parameters extracted from papers")

        return params

    def _merge_params(
        self,
        base: ExtractedParameters,
        candidate: ExtractedParameters,
    ) -> ExtractedParameters:
        """Merge candidate parameters into base using confidence scores."""
        # Base recovery rate
        base_conf = base.base_recovery_rate_confidence or 0.0
        cand_conf = candidate.base_recovery_rate_confidence or 0.0
        if candidate.base_recovery_rate is not None and cand_conf >= base_conf:
            base.base_recovery_rate = candidate.base_recovery_rate
            base.base_recovery_rate_confidence = candidate.base_recovery_rate_confidence
            base.base_recovery_rate_source = candidate.base_recovery_rate_source

        # Income thresholds (require both)
        base_conf = base.income_thresholds_confidence or 0.0
        cand_conf = candidate.income_thresholds_confidence or 0.0
        if (candidate.income_threshold_low is not None and
            candidate.income_threshold_high is not None and
            cand_conf >= base_conf):
            base.income_threshold_low = candidate.income_threshold_low
            base.income_threshold_high = candidate.income_threshold_high
            base.income_thresholds_confidence = candidate.income_thresholds_confidence
            base.income_thresholds_source = candidate.income_thresholds_source

        # Resilience thresholds (require both)
        base_conf = base.resilience_thresholds_confidence or 0.0
        cand_conf = candidate.resilience_thresholds_confidence or 0.0
        if (candidate.resilience_threshold_low is not None and
            candidate.resilience_threshold_high is not None and
            cand_conf >= base_conf):
            base.resilience_threshold_low = candidate.resilience_threshold_low
            base.resilience_threshold_high = candidate.resilience_threshold_high
            base.resilience_thresholds_confidence = candidate.resilience_thresholds_confidence
            base.resilience_thresholds_source = candidate.resilience_thresholds_source

        # Utility weights
        base_conf = base.utility_weights_confidence or 0.0
        cand_conf = candidate.utility_weights_confidence or 0.0
        if (candidate.utility_weights_confidence is not None and
            cand_conf >= base_conf):
            if candidate.utility_weight_self is not None:
                base.utility_weight_self = candidate.utility_weight_self
            if candidate.utility_weight_neighbor is not None:
                base.utility_weight_neighbor = candidate.utility_weight_neighbor
            if candidate.utility_weight_infrastructure is not None:
                base.utility_weight_infrastructure = candidate.utility_weight_infrastructure
            if candidate.utility_weight_business is not None:
                base.utility_weight_business = candidate.utility_weight_business
            base.utility_weights_confidence = candidate.utility_weights_confidence
            base.utility_weights_source = candidate.utility_weights_source

        return base

    def _validate_param(self, param_name: str, value: float | None) -> float | None:
        """Validate a parameter value against defined bounds."""
        if value is None:
            return None

        rules = PARAMETER_VALIDATION_RULES.get(param_name)
        if rules is None:
            return value

        if value < rules['min'] or value > rules['max']:
            logger.warning(
                f"Parameter {param_name}={value} out of bounds "
                f"[{rules['min']}, {rules['max']}], rejecting"
            )
            return None

        return value


class RecovUSParameterExtractor:
    """
    Extracts RecovUS-specific parameters from paper text using LLM.

    This extractor focuses on parameters specific to the RecovUS decision model:
    - Perception distribution (ASNA Index)
    - Community adequacy thresholds
    - State transition probabilities
    - Financial parameters

    The prompt template is designed based on skills.md guidance for
    disaster-specific parameter extraction.
    """

    PROMPT_TEMPLATE = """You are extracting household disaster recovery parameters from academic research papers.
The RecovUS model simulates household recovery decisions using these parameter categories:

## PARAMETER CATEGORIES

### 1. PERCEPTION DISTRIBUTION (ASNA Index)
What factors do households prioritize when deciding whether to rebuild?
- perception_infrastructure: % who prioritize infrastructure (power, water, roads)
- perception_social: % who prioritize social networks (neighbors rebuilding)
- perception_community: % who prioritize community assets (businesses, schools, amenities)
NOTE: These three values MUST sum to 1.0 (100%)

### 2. ADEQUACY THRESHOLDS
At what recovery level do households feel the community is "adequate" for rebuilding?
- adequacy_infrastructure: Infrastructure functionality threshold (0.0-1.0)
- adequacy_neighbor: Neighbor recovery threshold (0.0-1.0)
- adequacy_community_assets: Business/amenity availability threshold (0.0-1.0)

### 3. TRANSITION PROBABILITIES
What is the probability of starting repairs under different conditions?
- transition_r0: Probability of repair when ONLY financially feasible (community not ready)
- transition_r1: Probability of repair when BOTH financially feasible AND community adequate
- transition_r2: Probability of completing repairs once started (when community adequate)
- transition_relocate: Probability of relocating when financially infeasible

### 4. FINANCIAL PARAMETERS
What are typical financial assistance rates and amounts?
- insurance_penetration_rate: % of households with insurance coverage
- fema_ha_average: Average FEMA Housing Assistance grant (USD)
- sba_uptake_rate: % of eligible households who take SBA loans

## OUTPUT FORMAT

Return a JSON object with the following structure. Use null for parameters not found in the text.
Include exact quotes from the paper and confidence scores (0.0-1.0).

{{
  "disaster_type": "<flood|hurricane|earthquake|wildfire|tornado|other|null>",
  "disaster_event": "<specific event name if mentioned, or null>",

  "perception_distribution": {{
    "infrastructure": <float 0.0-1.0 or null>,
    "social": <float 0.0-1.0 or null>,
    "community": <float 0.0-1.0 or null>,
    "source_quote": "<exact quote supporting this or null>",
    "confidence": <0.0-1.0>
  }},

  "adequacy_thresholds": {{
    "infrastructure": <float 0.0-1.0 or null>,
    "neighbor": <float 0.0-1.0 or null>,
    "community_assets": <float 0.0-1.0 or null>,
    "source_quote": "<exact quote or null>",
    "confidence": <0.0-1.0>
  }},

  "transition_probabilities": {{
    "r0": <float 0.0-1.0 or null>,
    "r1": <float 0.0-1.0 or null>,
    "r2": <float 0.0-1.0 or null>,
    "relocate": <float 0.0-1.0 or null>,
    "source_quote": "<exact quote or null>",
    "confidence": <0.0-1.0>
  }},

  "financial_parameters": {{
    "insurance_penetration_rate": <float 0.0-1.0 or null>,
    "fema_ha_average": <float USD or null>,
    "sba_uptake_rate": <float 0.0-1.0 or null>,
    "source_quote": "<exact quote or null>",
    "confidence": <0.0-1.0>
  }}
}}

## EXTRACTION GUIDELINES

1. ONLY extract values EXPLICITLY stated or clearly implied in the text
2. Convert percentages to decimals (e.g., "65%" becomes 0.65)
3. If a timeline is given (e.g., "18 months to recovery"), compute monthly rate
4. Include EXACT quotes that support each extraction
5. Confidence scoring:
   - 0.9-1.0: Direct quote with numeric value
   - 0.7-0.9: Clear implication from text
   - 0.5-0.7: Reasonable inference
   - Below 0.5: Use null instead
6. Perception distribution MUST sum to 1.0 - adjust proportionally if needed
7. If only partial data available, extract what you can with appropriate confidence

## KEYWORD HINTS

- Perception: "prioritize", "consider", "decide based on", "most important factor"
- Adequacy: "threshold", "trigger", "once X% had", "tipping point"
- Transitions: "probability", "likelihood", "% who", "rate of repair"
- Financial: "insurance", "FEMA", "SBA", "assistance", "grant"

Document text to analyze:
{texts}"""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.05,
        max_tokens: int = 2500,
        chunk_size_chars: int = 8000,
        chunk_overlap_chars: int = 800,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def extract(self, papers: list[Paper]) -> RecovUSExtractedParameters:
        """
        Extract RecovUS parameters from paper text.

        Args:
            papers: List of papers with text content

        Returns:
            RecovUSExtractedParameters with any values found
        """
        if not papers:
            logger.warning("No papers provided for RecovUS parameter extraction")
            return RecovUSExtractedParameters()

        if not self.api_key:
            logger.warning("No Groq API key - cannot extract RecovUS parameters")
            return RecovUSExtractedParameters()

        merged = RecovUSExtractedParameters()

        for paper in papers:
            text = paper.abstract or ""
            if not text.strip():
                continue

            chunks = _chunk_text(text, self.chunk_size_chars, self.chunk_overlap_chars)
            if len(chunks) > 1:
                logger.info(
                    f"Split '{paper.title[:60]}' into {len(chunks)} chunks for RecovUS extraction"
                )

            for chunk in chunks:
                formatted = f"Title: {paper.title}\nText: {chunk}"
                prompt = self.PROMPT_TEMPLATE.format(texts=formatted)

                raw_params = self._call_llm(prompt)
                candidate = self._validate_and_convert(raw_params, log=False)
                merged = self._merge_params(merged, candidate)

        if merged.has_any_parameters():
            logger.info(f"Extracted RecovUS parameters: {merged.to_dict()}")
        else:
            logger.info("No RecovUS parameters extracted from papers")

        return merged

    def _call_llm(self, prompt: str) -> dict:
        """Call Groq LLM and parse response."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            logger.error("langchain-groq package not installed")
            return {}

        import os
        os.environ["GROQ_API_KEY"] = self.api_key

        llm = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            # Try structured output
            structured_llm = llm.with_structured_output(method="json_mode")
            response = structured_llm.invoke(prompt)

            if isinstance(response, dict):
                return response
        except Exception as e:
            logger.warning(f"Structured mode failed for RecovUS parameters: {e}")

        # Fallback to raw parsing
        try:
            raw_response = llm.invoke(prompt).content.strip()
            return self._parse_raw_response(raw_response)
        except Exception as e:
            logger.error(f"RecovUS parameter extraction LLM call failed: {e}")
            return {}

    def _parse_raw_response(self, raw: str) -> dict:
        """Parse raw LLM response."""
        cleaned = raw

        # Remove markdown fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        # Find JSON object
        start_idx = cleaned.find("{")
        if start_idx != -1:
            cleaned = cleaned[start_idx:]

        end_idx = cleaned.rfind("}")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"RecovUS parameter JSON parse error: {e}")
            return {}

    def _validate_and_convert(self, raw: dict, log: bool = True) -> RecovUSExtractedParameters:
        """Validate extracted values and convert to RecovUSExtractedParameters."""
        params = RecovUSExtractedParameters()

        # Extract disaster context (for logging/metadata)
        disaster_type = raw.get('disaster_type')
        disaster_event = raw.get('disaster_event')
        if disaster_type:
            params.disaster_type = disaster_type
            logger.info(f"Extracted parameters for disaster type: {disaster_type}")
        if disaster_event:
            params.disaster_event = disaster_event
            logger.info(f"Disaster event: {disaster_event}")

        # Perception distribution
        if raw.get('perception_distribution'):
            pct = raw['perception_distribution']
            infra = pct.get('infrastructure')
            social = pct.get('social')
            community = pct.get('community')
            confidence = pct.get('confidence', 0.5)

            # Validate and normalize to sum to 1.0
            if infra is not None and social is not None and community is not None:
                total = infra + social + community
                if total > 0:
                    # Normalize
                    infra = infra / total
                    social = social / total
                    community = community / total

                    # Validate bounds
                    if all(0 <= v <= 1 for v in [infra, social, community]):
                        params.perception_infrastructure = infra
                        params.perception_social = social
                        params.perception_community = community
                        params.perception_confidence = confidence
                        params.perception_source = pct.get('source_quote', 'Extracted from text')

        # Adequacy thresholds
        if raw.get('adequacy_thresholds'):
            adq = raw['adequacy_thresholds']
            confidence = adq.get('confidence', 0.5)

            infra = self._validate_threshold('adequacy_infrastructure', adq.get('infrastructure'))
            neighbor = self._validate_threshold('adequacy_neighbor', adq.get('neighbor'))
            community = self._validate_threshold('adequacy_community_assets', adq.get('community_assets'))

            if any(v is not None for v in [infra, neighbor, community]):
                if infra is not None:
                    params.adequacy_infrastructure = infra
                if neighbor is not None:
                    params.adequacy_neighbor = neighbor
                if community is not None:
                    params.adequacy_community_assets = community
                params.adequacy_confidence = confidence
                params.adequacy_source = adq.get('source_quote', 'Extracted from text')

        # Transition probabilities
        if raw.get('transition_probabilities'):
            trans = raw['transition_probabilities']
            confidence = trans.get('confidence', 0.5)

            r0 = self._validate_transition('transition_r0', trans.get('r0'))
            r1 = self._validate_transition('transition_r1', trans.get('r1'))
            r2 = self._validate_transition('transition_r2', trans.get('r2'))
            relocate = self._validate_transition('transition_relocate', trans.get('relocate'))

            if any(v is not None for v in [r0, r1, r2, relocate]):
                if r0 is not None:
                    params.transition_r0 = r0
                if r1 is not None:
                    params.transition_r1 = r1
                if r2 is not None:
                    params.transition_r2 = r2
                if relocate is not None:
                    params.transition_relocate = relocate
                params.transition_confidence = confidence
                params.transition_source = trans.get('source_quote', 'Extracted from text')

        # Financial parameters
        if raw.get('financial_parameters'):
            fin = raw['financial_parameters']
            confidence = fin.get('confidence', 0.5)

            insurance = self._validate_rate('insurance_penetration_rate', fin.get('insurance_penetration_rate'))
            sba = self._validate_rate('sba_uptake_rate', fin.get('sba_uptake_rate'))
            fema = fin.get('fema_ha_average')

            if any(v is not None for v in [insurance, sba, fema]):
                if insurance is not None:
                    params.insurance_penetration_rate = insurance
                if sba is not None:
                    params.sba_uptake_rate = sba
                if fema is not None and 5000 <= fema <= 100000:
                    params.fema_ha_average = fema
                params.financial_confidence = confidence
                params.financial_source = fin.get('source_quote', 'Extracted from text')

        if log:
            if params.has_any_parameters():
                logger.info(f"Extracted RecovUS parameters: {params.to_dict()}")
            else:
                logger.info("No RecovUS parameters extracted from papers")

        return params

    def _merge_params(
        self,
        base: RecovUSExtractedParameters,
        candidate: RecovUSExtractedParameters,
    ) -> RecovUSExtractedParameters:
        """Merge candidate parameters into base using confidence scores."""
        # Disaster context
        if candidate.disaster_type and not base.disaster_type:
            base.disaster_type = candidate.disaster_type
        if candidate.disaster_event and not base.disaster_event:
            base.disaster_event = candidate.disaster_event

        # Perception distribution
        base_conf = base.perception_confidence or 0.0
        cand_conf = candidate.perception_confidence or 0.0
        if (candidate.perception_infrastructure is not None and
            cand_conf >= base_conf):
            base.perception_infrastructure = candidate.perception_infrastructure
            base.perception_social = candidate.perception_social
            base.perception_community = candidate.perception_community
            base.perception_confidence = candidate.perception_confidence
            base.perception_source = candidate.perception_source

        # Adequacy thresholds
        base_conf = base.adequacy_confidence or 0.0
        cand_conf = candidate.adequacy_confidence or 0.0
        if (candidate.adequacy_confidence is not None and
            cand_conf >= base_conf):
            if candidate.adequacy_infrastructure is not None:
                base.adequacy_infrastructure = candidate.adequacy_infrastructure
            if candidate.adequacy_neighbor is not None:
                base.adequacy_neighbor = candidate.adequacy_neighbor
            if candidate.adequacy_community_assets is not None:
                base.adequacy_community_assets = candidate.adequacy_community_assets
            base.adequacy_confidence = candidate.adequacy_confidence
            base.adequacy_source = candidate.adequacy_source

        # Transition probabilities
        base_conf = base.transition_confidence or 0.0
        cand_conf = candidate.transition_confidence or 0.0
        if (candidate.transition_confidence is not None and
            cand_conf >= base_conf):
            if candidate.transition_r0 is not None:
                base.transition_r0 = candidate.transition_r0
            if candidate.transition_r1 is not None:
                base.transition_r1 = candidate.transition_r1
            if candidate.transition_r2 is not None:
                base.transition_r2 = candidate.transition_r2
            if candidate.transition_relocate is not None:
                base.transition_relocate = candidate.transition_relocate
            base.transition_confidence = candidate.transition_confidence
            base.transition_source = candidate.transition_source

        # Financial parameters
        base_conf = base.financial_confidence or 0.0
        cand_conf = candidate.financial_confidence or 0.0
        if (candidate.financial_confidence is not None and
            cand_conf >= base_conf):
            if candidate.insurance_penetration_rate is not None:
                base.insurance_penetration_rate = candidate.insurance_penetration_rate
            if candidate.fema_ha_average is not None:
                base.fema_ha_average = candidate.fema_ha_average
            if candidate.sba_uptake_rate is not None:
                base.sba_uptake_rate = candidate.sba_uptake_rate
            base.financial_confidence = candidate.financial_confidence
            base.financial_source = candidate.financial_source

        return base

    def _validate_threshold(self, param_name: str, value: float | None) -> float | None:
        """Validate an adequacy threshold."""
        if value is None:
            return None

        rules = RECOVUS_VALIDATION_RULES.get(param_name)
        if rules is None:
            # Default bounds for thresholds
            if 0.0 <= value <= 1.0:
                return value
            return None

        if value < rules['min'] or value > rules['max']:
            logger.warning(f"RecovUS {param_name}={value} out of bounds, rejecting")
            return None

        return value

    def _validate_transition(self, param_name: str, value: float | None) -> float | None:
        """Validate a transition probability."""
        if value is None:
            return None

        rules = RECOVUS_VALIDATION_RULES.get(param_name)
        if rules is None:
            # Default bounds for probabilities
            if 0.0 <= value <= 1.0:
                return value
            return None

        if value < rules['min'] or value > rules['max']:
            logger.warning(f"RecovUS {param_name}={value} out of bounds, rejecting")
            return None

        return value

    def _validate_rate(self, param_name: str, value: float | None) -> float | None:
        """Validate a rate parameter."""
        if value is None:
            return None

        # Rates should be between 0 and 1
        if 0.0 <= value <= 1.0:
            return value

        logger.warning(f"RecovUS {param_name}={value} not a valid rate, rejecting")
        return None


def get_fallback_heuristics(use_recovus: bool = False) -> list[Heuristic]:
    """
    Return static fallback heuristics when LLM extraction fails.

    These are based on common disaster recovery research findings.

    Args:
        use_recovus: If True, return RecovUS-style heuristics that modify
                     transition probabilities instead of boost/extra_recovery.
    """
    if use_recovus:
        return get_recovus_fallback_heuristics()

    fallback_data = [
        {
            'condition': "ctx['avg_neighbor_recovery'] > 0.5",
            'action': {'boost': 1.5},
            'source': 'Neighbor influence'
        },
        {
            'condition': "ctx['avg_infra_func'] < 0.3",
            'action': {'boost': 0.6},
            'source': 'Infrastructure barriers'
        },
        {
            'condition': "ctx['avg_business_avail'] > 0.6",
            'action': {'boost': 1.25},
            'source': 'Economic incentive'
        },
        {
            'condition': "ctx['num_neighbors'] > 4",
            'action': {'extra_recovery': 0.08},
            'source': 'Network cohesion'
        },
        {
            'condition': "ctx['resilience'] > 0.7",
            'action': {'boost': 1.3},
            'source': 'High resilience'
        },
        {
            'condition': "ctx['income_level'] == 'low' and ctx['avg_infra_func'] < 0.4",
            'action': {'boost': 0.5},
            'source': 'Vulnerability compound'
        },
    ]

    return [
        Heuristic(
            condition_str=h['condition'],
            action=h['action'],
            source=h['source']
        ).compile()
        for h in fallback_data
    ]


def get_recovus_fallback_heuristics() -> list[Heuristic]:
    """
    Return RecovUS-style fallback heuristics.

    These heuristics modify transition probabilities (r0, r1, r2) and
    adequacy thresholds instead of boost/extra_recovery.
    """
    recovus_fallback_data = [
        # Social network effects
        {
            'condition': "ctx['perception_type'] == 'social' and ctx['avg_neighbor_recovery'] > 0.5",
            'action': {'modify_r1': 1.15, 'modify_adq_nbr': -0.05},
            'source': 'Social capital facilitates recovery'
        },
        {
            'condition': "ctx['perception_type'] == 'social' and ctx['avg_neighbor_recovery'] < 0.3",
            'action': {'modify_r1': 0.85, 'modify_r0': 0.7},
            'source': 'Low neighbor recovery discourages repair'
        },

        # Infrastructure effects
        {
            'condition': "ctx['perception_type'] == 'infrastructure' and ctx['avg_infra_func'] > 0.6",
            'action': {'modify_r1': 1.2, 'modify_adq_infr': -0.1},
            'source': 'Infrastructure restoration enables recovery'
        },
        {
            'condition': "ctx['avg_infra_func'] < 0.3",
            'action': {'modify_r1': 0.7, 'modify_r0': 0.5},
            'source': 'Poor infrastructure impedes all recovery'
        },

        # Economic/income effects
        {
            'condition': "ctx['income_level'] == 'low' and ctx['is_feasible']",
            'action': {'modify_r1': 0.9, 'modify_r0': 0.8},
            'source': 'Low-income households face barriers even when feasible'
        },
        {
            'condition': "ctx['income_level'] == 'high' and ctx['is_feasible']",
            'action': {'modify_r1': 1.1, 'modify_r0': 1.2},
            'source': 'High-income households recover faster'
        },

        # Damage severity effects
        {
            'condition': "ctx['damage_severity'] == 'severe' or ctx['damage_severity'] == 'destroyed'",
            'action': {'modify_r1': 0.85, 'modify_r2': 0.8},
            'source': 'Severe damage slows recovery progress'
        },

        # Community assets effects
        {
            'condition': "ctx['perception_type'] == 'community' and ctx['avg_business_avail'] > 0.5",
            'action': {'modify_r1': 1.15, 'modify_adq_cas': -0.1},
            'source': 'Business availability supports community-focused households'
        },

        # Resilience effects
        {
            'condition': "ctx['resilience'] > 0.7",
            'action': {'modify_r1': 1.1, 'modify_r2': 1.1},
            'source': 'High resilience accelerates recovery decisions'
        },
    ]

    return [
        Heuristic(
            condition_str=h['condition'],
            action=h['action'],
            source=h['source']
        ).compile()
        for h in recovus_fallback_data
    ]


def _heuristics_signature(heuristics: list[Heuristic]) -> list[tuple[str, tuple[tuple[str, Any], ...], str]]:
    """Create a stable signature for comparing heuristic sets."""
    return sorted(
        (h.condition_str, tuple(sorted(h.action.items())), h.source)
        for h in heuristics
    )


def _dedupe_heuristics(heuristics: list[Heuristic]) -> list[Heuristic]:
    """Remove duplicate heuristics by condition, action, and source."""
    seen: set[tuple[str, tuple[tuple[str, Any], ...], str]] = set()
    deduped: list[Heuristic] = []
    for h in heuristics:
        key = (h.condition_str, tuple(sorted(h.action.items())), h.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks for LLM processing."""
    if not text:
        return []
    if max_chars <= 0:
        return [text]

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    overlap = max(0, min(overlap, max_chars - 1))
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


def _is_fallback_heuristics(heuristics: list[Heuristic], use_recovus: bool = False) -> bool:
    """Check if heuristics match the built-in fallback set."""
    if not heuristics:
        return True
    fallback = get_fallback_heuristics(use_recovus=use_recovus)
    return _heuristics_signature(heuristics) == _heuristics_signature(fallback)


def build_knowledge_base(
    serpapi_key: str,
    groq_api_key: str,
    query: str = "heuristics in agent-based models for community disaster recovery",
    num_papers: int = 5,
    cache_dir: Path | None = None,
    use_recovus: bool = False,
    us_only: bool = False,
) -> list[Heuristic]:
    """
    Build a knowledge base of heuristics from academic research.

    This is the main entry point for the RAG pipeline:
    1. Search Google Scholar for relevant papers
    2. Extract heuristics using LLM
    3. Fall back to static heuristics if needed

    Args:
        serpapi_key: API key for SerpApi
        groq_api_key: API key for Groq
        query: Search query for Google Scholar
        num_papers: Number of papers to retrieve
        cache_dir: Directory to cache API results
        use_recovus: If True, extract RecovUS-style heuristics
        us_only: If True, bias Scholar query toward US-based studies

    Returns:
        List of compiled Heuristic objects
    """
    # Retrieve papers
    retriever = ScholarRetriever(serpapi_key, cache_dir)
    if us_only:
        query_lower = query.lower()
        if "united states" not in query_lower and "u.s." not in query_lower and "usa" not in query_lower:
            query = f"{query} United States"
    papers = retriever.search(query, num_papers)

    if papers:
        logger.info(f"Retrieved {len(papers)} papers:")
        for i, p in enumerate(papers, 1):
            logger.info(f"  {i}. {p.title} ({p.year}) - cited by {p.cited_by}")

    # Extract heuristics
    if papers:
        extractor_cls = RecovUSHeuristicExtractor if use_recovus else HeuristicExtractor
        extractor = extractor_cls(groq_api_key)
        heuristics = extractor.extract(papers)

        if heuristics:
            logger.info(f"Extracted {len(heuristics)} heuristics from research")
            return heuristics

    # Fallback
    logger.info("Using fallback heuristics")
    return get_fallback_heuristics(use_recovus=use_recovus)


def build_knowledge_base_from_pdfs(
    pdf_dir: Path | str,
    groq_api_key: str,
    keywords: list[str] | None = None,
    num_papers: int = 5,
    use_recovus: bool = False,
    us_only: bool = False,
    use_full_text: bool = True,
    pdf_max_pages: int | None = None,
) -> list[Heuristic]:
    """
    Build a knowledge base from local PDF files.

    This allows using your own research library instead of Google Scholar.

    Args:
        pdf_dir: Directory containing PDF files
        groq_api_key: API key for Groq LLM
        keywords: Optional keywords to filter PDFs by filename
        num_papers: Maximum number of PDFs to process
        use_recovus: If True, extract RecovUS-style heuristics
        us_only: If True, only include US-based studies
        use_full_text: If True, extract from full document text
        pdf_max_pages: Maximum pages per PDF to read (None = all pages)

    Returns:
        List of compiled Heuristic objects
    """
    from .pdf_retrieval import LocalPaperRetriever, DISASTER_RECOVERY_KEYWORDS

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return get_fallback_heuristics(use_recovus=use_recovus)

    # Use default disaster recovery keywords if none provided
    if keywords is None:
        keywords = DISASTER_RECOVERY_KEYWORDS

    # Load papers from PDFs
    retriever = LocalPaperRetriever(pdf_dir)
    local_papers = retriever.load_papers(
        keywords=keywords,
        max_papers=num_papers,
        max_pages=pdf_max_pages,
        us_only=us_only,
    )

    if not local_papers:
        logger.warning("No papers loaded from PDFs")
        return get_fallback_heuristics(use_recovus=use_recovus)

    logger.info(f"Loaded {len(local_papers)} papers from PDFs:")
    for i, p in enumerate(local_papers, 1):
        logger.info(f"  {i}. {p.title[:60]}...")
    if use_full_text:
        logger.info("Using full document text for extraction")

    # Convert to Paper format for the extractor
    papers = []
    for p in local_papers:
        text = p.full_text if use_full_text and p.full_text else p.abstract
        papers.append(
            Paper(
                title=p.title,
                abstract=text,
                authors=p.authors,
                year=p.year,
                link=str(p.filepath),
                cited_by=0,
            )
        )

    # Extract heuristics
    if not groq_api_key:
        logger.warning("No Groq API key - cannot extract heuristics from PDFs")
        return get_fallback_heuristics(use_recovus=use_recovus)

    extractor_cls = RecovUSHeuristicExtractor if use_recovus else HeuristicExtractor
    extractor = extractor_cls(groq_api_key)
    heuristics = extractor.extract(papers)

    if heuristics:
        logger.info(f"Extracted {len(heuristics)} heuristics from local PDFs")
        return heuristics

    logger.info("No heuristics extracted, using fallback")
    return get_fallback_heuristics(use_recovus=use_recovus)


def build_knowledge_base_hybrid(
    pdf_dir: Path | str | None = None,
    serpapi_key: str = "",
    groq_api_key: str = "",
    scholar_query: str = "disaster recovery heuristics agent-based model",
    num_papers: int = 5,
    prefer_local: bool = True,
    use_recovus: bool = False,
    us_only: bool = False,
    use_full_text: bool = True,
    pdf_max_pages: int | None = None,
) -> list[Heuristic]:
    """
    Build knowledge base from both local PDFs and Google Scholar.

    Args:
        pdf_dir: Directory containing local PDFs (optional)
        serpapi_key: SerpApi key for Google Scholar
        groq_api_key: Groq API key for LLM
        scholar_query: Query for Google Scholar search
        num_papers: Max papers from each source
        prefer_local: If True, use local PDFs first; if False, use Scholar first
        use_recovus: If True, extract RecovUS-style heuristics
        us_only: If True, only include US-based studies
        use_full_text: If True, extract from full document text
        pdf_max_pages: Maximum pages per PDF to read (None = all pages)

    Returns:
        List of compiled Heuristic objects
    """
    heuristics = []

    # Try local PDFs first if preferred
    if prefer_local and pdf_dir:
        logger.info("Attempting to load heuristics from local PDFs...")
        heuristics = build_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_api_key,
            num_papers=num_papers,
            use_recovus=use_recovus,
            us_only=us_only,
            use_full_text=use_full_text,
            pdf_max_pages=pdf_max_pages,
        )
        # Check if we got non-fallback heuristics
        if heuristics and not _is_fallback_heuristics(heuristics, use_recovus=use_recovus):
            return heuristics

    # Try Google Scholar
    if serpapi_key:
        logger.info("Attempting to load heuristics from Google Scholar...")
        heuristics = build_knowledge_base(
            serpapi_key=serpapi_key,
            groq_api_key=groq_api_key,
            query=scholar_query,
            num_papers=num_papers,
            use_recovus=use_recovus,
            us_only=us_only,
        )
        if heuristics and not _is_fallback_heuristics(heuristics, use_recovus=use_recovus):
            return heuristics

    # Try local PDFs if not preferred but Scholar failed
    if not prefer_local and pdf_dir:
        logger.info("Scholar failed, trying local PDFs...")
        heuristics = build_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_api_key,
            num_papers=num_papers,
            use_recovus=use_recovus,
            us_only=us_only,
            use_full_text=use_full_text,
            pdf_max_pages=pdf_max_pages,
        )
        if heuristics:
            return heuristics

    # Final fallback
    return get_fallback_heuristics(use_recovus=use_recovus)


# =============================================================================
# Combined Knowledge Base with RecovUS Parameter Extraction
# =============================================================================

@dataclass
class KnowledgeBaseResult:
    """
    Combined result from RAG pipeline containing both heuristics and RecovUS parameters.

    This enables the simulation to use:
    1. Behavioral heuristics that modify transition probabilities
    2. RecovUS parameters extracted from research (with confidence scores)
    """
    heuristics: list[Heuristic]
    recovus_params: RecovUSExtractedParameters
    papers_processed: int = 0

    def has_recovus_params(self) -> bool:
        """Check if any RecovUS parameters were extracted."""
        return self.recovus_params.has_any_parameters()

    def summary(self) -> str:
        """Get a summary of what was extracted."""
        lines = [
            f"Papers processed: {self.papers_processed}",
            f"Heuristics extracted: {len(self.heuristics)}",
        ]

        if self.has_recovus_params():
            lines.append("RecovUS parameters extracted:")
            p = self.recovus_params
            if p.perception_infrastructure is not None:
                lines.append(f"  - Perception: infra={p.perception_infrastructure:.0%}, "
                           f"social={p.perception_social:.0%}, "
                           f"community={p.perception_community:.0%}")
            if p.adequacy_infrastructure is not None:
                lines.append(f"  - Adequacy: infra={p.adequacy_infrastructure:.0%}, "
                           f"neighbor={p.adequacy_neighbor:.0%}, "
                           f"community={p.adequacy_community_assets:.0%}")
            if p.transition_r0 is not None:
                lines.append(f"  - Transitions: r0={p.transition_r0:.0%}, "
                           f"r1={p.transition_r1:.0%}, r2={p.transition_r2:.0%}")
        else:
            lines.append("No RecovUS parameters extracted (using defaults)")

        return "\n".join(lines)


def build_full_knowledge_base(
    serpapi_key: str = "",
    groq_api_key: str = "",
    query: str = "heuristics in agent-based models for community disaster recovery",
    num_papers: int = 5,
    cache_dir: Path | None = None,
    extract_recovus: bool = True,
    use_recovus_heuristics: bool = False,
    us_only: bool = False,
) -> KnowledgeBaseResult:
    """
    Build a complete knowledge base with both heuristics and RecovUS parameters.

    This is the recommended entry point for the full RAG pipeline, extracting:
    1. Behavioral heuristics (modify transition probabilities)
    2. RecovUS parameters (perception distribution, adequacy thresholds, etc.)

    Args:
        serpapi_key: API key for SerpApi (Google Scholar)
        groq_api_key: API key for Groq LLM
        query: Search query for Google Scholar
        num_papers: Number of papers to retrieve
        cache_dir: Directory to cache API results
        extract_recovus: Whether to also extract RecovUS parameters
        use_recovus_heuristics: If True, extract RecovUS-style heuristics
        us_only: If True, bias Scholar query toward US-based studies

    Returns:
        KnowledgeBaseResult containing heuristics and RecovUS parameters
    """
    papers: list[Paper] = []
    heuristics: list[Heuristic] = []
    recovus_params = RecovUSExtractedParameters()

    # Retrieve papers
    if serpapi_key:
        retriever = ScholarRetriever(serpapi_key, cache_dir)
        if us_only:
            query_lower = query.lower()
            if "united states" not in query_lower and "u.s." not in query_lower and "usa" not in query_lower:
                query = f"{query} United States"
        papers = retriever.search(query, num_papers)

        if papers:
            logger.info(f"Retrieved {len(papers)} papers for knowledge base")

    # Extract heuristics
    if papers and groq_api_key:
        extractor_cls = RecovUSHeuristicExtractor if use_recovus_heuristics else HeuristicExtractor
        extractor = extractor_cls(groq_api_key)
        heuristics = extractor.extract(papers)

        if heuristics:
            logger.info(f"Extracted {len(heuristics)} heuristics from research")

    # Extract RecovUS parameters
    if papers and groq_api_key and extract_recovus:
        try:
            recovus_extractor = RecovUSParameterExtractor(groq_api_key)
            recovus_params = recovus_extractor.extract(papers)

            if recovus_params.has_any_parameters():
                logger.info("Extracted RecovUS parameters from research")
                if recovus_params.disaster_type:
                    logger.debug(f"  Disaster type: {recovus_params.disaster_type}")
                if recovus_params.perception_infrastructure is not None:
                    logger.debug(f"  Perception distribution: "
                               f"{recovus_params.perception_infrastructure:.0%}/"
                               f"{recovus_params.perception_social:.0%}/"
                               f"{recovus_params.perception_community:.0%}")
        except Exception as e:
            logger.warning(f"Failed to extract RecovUS parameters: {e}")
            recovus_params = RecovUSExtractedParameters()

    # Fallback heuristics if none extracted
    if not heuristics:
        logger.info("Using fallback heuristics")
        heuristics = get_fallback_heuristics(use_recovus=use_recovus_heuristics)

    return KnowledgeBaseResult(
        heuristics=heuristics,
        recovus_params=recovus_params,
        papers_processed=len(papers),
    )


def build_full_knowledge_base_from_pdfs(
    pdf_dir: Path | str,
    groq_api_key: str,
    keywords: list[str] | None = None,
    num_papers: int = 5,
    extract_recovus: bool = True,
    use_recovus_heuristics: bool = False,
    us_only: bool = False,
    use_full_text: bool = True,
    pdf_max_pages: int | None = None,
) -> KnowledgeBaseResult:
    """
    Build a complete knowledge base from local PDFs.

    Args:
        pdf_dir: Directory containing PDF files
        groq_api_key: API key for Groq LLM
        keywords: Optional keywords to filter PDFs by filename
        num_papers: Maximum number of PDFs to process
        extract_recovus: Whether to also extract RecovUS parameters
        use_recovus_heuristics: If True, extract RecovUS-style heuristics
        us_only: If True, only include US-based studies
        use_full_text: If True, extract from full document text
        pdf_max_pages: Maximum pages per PDF to read (None = all pages)

    Returns:
        KnowledgeBaseResult containing heuristics and RecovUS parameters
    """
    from .pdf_retrieval import LocalPaperRetriever, DISASTER_RECOVERY_KEYWORDS

    pdf_dir = Path(pdf_dir)
    heuristics: list[Heuristic] = []
    recovus_params = RecovUSExtractedParameters()
    papers: list[Paper] = []

    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return KnowledgeBaseResult(
            heuristics=get_fallback_heuristics(use_recovus=use_recovus_heuristics),
            recovus_params=recovus_params,
            papers_processed=0,
        )

    # Use default disaster recovery keywords if none provided
    if keywords is None:
        keywords = DISASTER_RECOVERY_KEYWORDS

    # Load papers from PDFs
    retriever = LocalPaperRetriever(pdf_dir)
    local_papers = retriever.load_papers(
        keywords=keywords,
        max_papers=num_papers,
        max_pages=pdf_max_pages,
        us_only=us_only,
    )

    if not local_papers:
        logger.warning("No papers loaded from PDFs")
        return KnowledgeBaseResult(
            heuristics=get_fallback_heuristics(use_recovus=use_recovus_heuristics),
            recovus_params=recovus_params,
            papers_processed=0,
        )

    logger.info(f"Loaded {len(local_papers)} papers from PDFs")
    if use_full_text:
        logger.info("Using full document text for extraction")

    # Convert to Paper format
    papers = []
    for p in local_papers:
        text = p.full_text if use_full_text and p.full_text else p.abstract
        papers.append(
            Paper(
                title=p.title,
                abstract=text,
                authors=p.authors,
                year=p.year,
                link=str(p.filepath),
                cited_by=0,
            )
        )

    # Extract heuristics
    if groq_api_key:
        extractor_cls = RecovUSHeuristicExtractor if use_recovus_heuristics else HeuristicExtractor
        extractor = extractor_cls(groq_api_key)
        heuristics = extractor.extract(papers)

        if heuristics:
            logger.info(f"Extracted {len(heuristics)} heuristics from local PDFs")

    # Extract RecovUS parameters
    if groq_api_key and extract_recovus:
        try:
            recovus_extractor = RecovUSParameterExtractor(groq_api_key)
            recovus_params = recovus_extractor.extract(papers)

            if recovus_params.has_any_parameters():
                logger.info("Extracted RecovUS parameters from local PDFs")
        except Exception as e:
            logger.warning(f"Failed to extract RecovUS parameters: {e}")
            recovus_params = RecovUSExtractedParameters()

    # Fallback heuristics if none extracted
    if not heuristics:
        logger.info("Using fallback heuristics")
        heuristics = get_fallback_heuristics(use_recovus=use_recovus_heuristics)

    return KnowledgeBaseResult(
        heuristics=heuristics,
        recovus_params=recovus_params,
        papers_processed=len(papers),
    )


def build_full_knowledge_base_hybrid(
    pdf_dir: Path | str | None = None,
    serpapi_key: str = "",
    groq_api_key: str = "",
    scholar_query: str = "disaster recovery heuristics agent-based model",
    num_papers: int = 5,
    prefer_local: bool = True,
    extract_recovus: bool = True,
    use_recovus_heuristics: bool = False,
    us_only: bool = False,
    use_full_text: bool = True,
    pdf_max_pages: int | None = None,
) -> KnowledgeBaseResult:
    """
    Build a complete knowledge base from both local PDFs and Google Scholar.

    This is the most flexible entry point, combining multiple sources.

    Args:
        pdf_dir: Directory containing local PDFs (optional)
        serpapi_key: SerpApi key for Google Scholar
        groq_api_key: Groq API key for LLM
        scholar_query: Query for Google Scholar search
        num_papers: Max papers from each source
        prefer_local: If True, use local PDFs first
        extract_recovus: Whether to also extract RecovUS parameters
        use_recovus_heuristics: If True, extract RecovUS-style heuristics
        us_only: If True, only include US-based studies
        use_full_text: If True, extract from full document text
        pdf_max_pages: Maximum pages per PDF to read (None = all pages)

    Returns:
        KnowledgeBaseResult containing heuristics and RecovUS parameters
    """
    # Try preferred source first
    if prefer_local and pdf_dir:
        logger.info("Attempting to build knowledge base from local PDFs...")
        result = build_full_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_api_key,
            num_papers=num_papers,
            extract_recovus=extract_recovus,
            use_recovus_heuristics=use_recovus_heuristics,
            us_only=us_only,
            use_full_text=use_full_text,
            pdf_max_pages=pdf_max_pages,
        )
        # Check if we got non-fallback heuristics
        if result.heuristics and not _is_fallback_heuristics(result.heuristics, use_recovus=use_recovus_heuristics):
            return result

    # Try Google Scholar
    if serpapi_key:
        logger.info("Attempting to build knowledge base from Google Scholar...")
        result = build_full_knowledge_base(
            serpapi_key=serpapi_key,
            groq_api_key=groq_api_key,
            query=scholar_query,
            num_papers=num_papers,
            extract_recovus=extract_recovus,
            use_recovus_heuristics=use_recovus_heuristics,
            us_only=us_only,
        )
        if result.heuristics and not _is_fallback_heuristics(result.heuristics, use_recovus=use_recovus_heuristics):
            return result

    # Try non-preferred source if preferred failed
    if not prefer_local and pdf_dir:
        logger.info("Scholar failed, trying local PDFs...")
        result = build_full_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_api_key,
            num_papers=num_papers,
            extract_recovus=extract_recovus,
            use_recovus_heuristics=use_recovus_heuristics,
            us_only=us_only,
            use_full_text=use_full_text,
            pdf_max_pages=pdf_max_pages,
        )
        if result.heuristics:
            return result

    # Final fallback
    return KnowledgeBaseResult(
        heuristics=get_fallback_heuristics(use_recovus=use_recovus_heuristics),
        recovus_params=RecovUSExtractedParameters(),
        papers_processed=0,
    )
