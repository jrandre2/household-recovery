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
2. AUGMENT: Use the paper abstracts as context for the LLM
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
    """Extracts behavioral heuristics from paper abstracts using LLM."""

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

Abstracts:
{abstracts}"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.05, max_tokens: int = 1200):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, papers: list[Paper]) -> list[Heuristic]:
        """
        Extract heuristics from paper abstracts.

        Args:
            papers: List of papers with abstracts

        Returns:
            List of compiled Heuristic objects
        """
        if not papers:
            logger.warning("No papers provided for heuristic extraction")
            return []

        if not self.api_key:
            logger.warning("No Groq API key - cannot extract heuristics")
            return []

        # Format abstracts
        formatted = "\n\n".join(
            f"Title: {p.title}\nAbstract: {p.abstract}"
            for p in papers
        )
        prompt = self.PROMPT_TEMPLATE.format(abstracts=formatted)

        # Call LLM
        raw_heuristics = self._call_llm(prompt)

        # Compile and validate
        return self._compile_heuristics(raw_heuristics)

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

    def _compile_heuristics(self, raw_heuristics: list[dict]) -> list[Heuristic]:
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

        logger.info(f"Compiled {len(compiled)}/{len(raw_heuristics)} heuristics")
        return compiled


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


class ParameterExtractor:
    """
    Extracts numeric simulation parameters from paper abstracts using LLM.

    This is separate from heuristic extraction to avoid overloading a single prompt.
    The extracted parameters can be used to configure simulations based on
    empirical findings from disaster recovery literature.
    """

    PROMPT_TEMPLATE = """You are a research data extractor. Analyze these academic paper abstracts about disaster recovery and extract NUMERIC PARAMETERS mentioned in the research.

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

Abstracts to analyze:
{abstracts}"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.05, max_tokens: int = 2000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, papers: list[Paper]) -> ExtractedParameters:
        """
        Extract numeric parameters from paper abstracts.

        Args:
            papers: List of papers with abstracts

        Returns:
            ExtractedParameters with any values found
        """
        if not papers:
            logger.warning("No papers provided for parameter extraction")
            return ExtractedParameters()

        if not self.api_key:
            logger.warning("No Groq API key - cannot extract parameters")
            return ExtractedParameters()

        # Format abstracts
        formatted = "\n\n".join(
            f"Title: {p.title}\nAbstract: {p.abstract}"
            for p in papers
        )
        prompt = self.PROMPT_TEMPLATE.format(abstracts=formatted)

        # Call LLM
        raw_params = self._call_llm(prompt)

        # Validate and convert
        return self._validate_and_convert(raw_params)

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

    def _validate_and_convert(self, raw: dict) -> ExtractedParameters:
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
                    params.base_recovery_rate_source = brr.get('source_quote', 'Extracted from abstracts')

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
                    params.income_thresholds_source = inc.get('source_quote', 'Extracted from abstracts')

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
                    params.resilience_thresholds_source = res.get('source_quote', 'Extracted from abstracts')

        # Utility weights
        has_weights = False
        if raw.get('social_influence_weight'):
            siw = raw['social_influence_weight']
            value = self._validate_param('utility_weight_neighbor', siw.get('value'))
            if value is not None:
                params.utility_weight_neighbor = value
                params.utility_weights_confidence = siw.get('confidence', 0.5)
                params.utility_weights_source = siw.get('source_quote', 'Extracted from abstracts')
                has_weights = True

        if raw.get('infrastructure_weight'):
            iw = raw['infrastructure_weight']
            value = self._validate_param('utility_weight_infrastructure', iw.get('value'))
            if value is not None:
                params.utility_weight_infrastructure = value
                if not has_weights:
                    params.utility_weights_confidence = iw.get('confidence', 0.5)
                    params.utility_weights_source = iw.get('source_quote', 'Extracted from abstracts')

        if params.has_any_parameters():
            logger.info(f"Extracted parameters: {params.to_dict()}")
        else:
            logger.info("No numeric parameters extracted from papers")

        return params

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


def get_fallback_heuristics() -> list[Heuristic]:
    """
    Return static fallback heuristics when LLM extraction fails.

    These are based on common disaster recovery research findings.
    """
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


def build_knowledge_base(
    serpapi_key: str,
    groq_api_key: str,
    query: str = "heuristics in agent-based models for community disaster recovery",
    num_papers: int = 5,
    cache_dir: Path | None = None
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

    Returns:
        List of compiled Heuristic objects
    """
    # Retrieve papers
    retriever = ScholarRetriever(serpapi_key, cache_dir)
    papers = retriever.search(query, num_papers)

    if papers:
        logger.info(f"Retrieved {len(papers)} papers:")
        for i, p in enumerate(papers, 1):
            logger.info(f"  {i}. {p.title} ({p.year}) - cited by {p.cited_by}")

    # Extract heuristics
    if papers:
        extractor = HeuristicExtractor(groq_api_key)
        heuristics = extractor.extract(papers)

        if heuristics:
            logger.info(f"Extracted {len(heuristics)} heuristics from research")
            return heuristics

    # Fallback
    logger.info("Using fallback heuristics")
    return get_fallback_heuristics()


def build_knowledge_base_from_pdfs(
    pdf_dir: Path | str,
    groq_api_key: str,
    keywords: list[str] | None = None,
    num_papers: int = 5
) -> list[Heuristic]:
    """
    Build a knowledge base from local PDF files.

    This allows using your own research library instead of Google Scholar.

    Args:
        pdf_dir: Directory containing PDF files
        groq_api_key: API key for Groq LLM
        keywords: Optional keywords to filter PDFs by filename
        num_papers: Maximum number of PDFs to process

    Returns:
        List of compiled Heuristic objects
    """
    from .pdf_retrieval import LocalPaperRetriever, DISASTER_RECOVERY_KEYWORDS

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return get_fallback_heuristics()

    # Use default disaster recovery keywords if none provided
    if keywords is None:
        keywords = DISASTER_RECOVERY_KEYWORDS

    # Load papers from PDFs
    retriever = LocalPaperRetriever(pdf_dir)
    local_papers = retriever.load_papers(keywords=keywords, max_papers=num_papers)

    if not local_papers:
        logger.warning("No papers loaded from PDFs")
        return get_fallback_heuristics()

    logger.info(f"Loaded {len(local_papers)} papers from PDFs:")
    for i, p in enumerate(local_papers, 1):
        logger.info(f"  {i}. {p.title[:60]}...")

    # Convert to Paper format for the extractor
    papers = [
        Paper(
            title=p.title,
            abstract=p.abstract,
            authors=p.authors,
            year=p.year,
            link=str(p.filepath),
            cited_by=0
        )
        for p in local_papers
    ]

    # Extract heuristics
    if not groq_api_key:
        logger.warning("No Groq API key - cannot extract heuristics from PDFs")
        return get_fallback_heuristics()

    extractor = HeuristicExtractor(groq_api_key)
    heuristics = extractor.extract(papers)

    if heuristics:
        logger.info(f"Extracted {len(heuristics)} heuristics from local PDFs")
        return heuristics

    logger.info("No heuristics extracted, using fallback")
    return get_fallback_heuristics()


def build_knowledge_base_hybrid(
    pdf_dir: Path | str | None = None,
    serpapi_key: str = "",
    groq_api_key: str = "",
    scholar_query: str = "disaster recovery heuristics agent-based model",
    num_papers: int = 5,
    prefer_local: bool = True
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
            num_papers=num_papers
        )
        # Check if we got non-fallback heuristics
        if heuristics and any(h.source != 'Neighbor influence' for h in heuristics):
            return heuristics

    # Try Google Scholar
    if serpapi_key:
        logger.info("Attempting to load heuristics from Google Scholar...")
        heuristics = build_knowledge_base(
            serpapi_key=serpapi_key,
            groq_api_key=groq_api_key,
            query=scholar_query,
            num_papers=num_papers
        )
        if heuristics and any(h.source != 'Neighbor influence' for h in heuristics):
            return heuristics

    # Try local PDFs if not preferred but Scholar failed
    if not prefer_local and pdf_dir:
        logger.info("Scholar failed, trying local PDFs...")
        heuristics = build_knowledge_base_from_pdfs(
            pdf_dir=pdf_dir,
            groq_api_key=groq_api_key,
            num_papers=num_papers
        )
        if heuristics:
            return heuristics

    # Final fallback
    return get_fallback_heuristics()
