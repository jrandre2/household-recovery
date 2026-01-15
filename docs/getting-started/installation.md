# Installation

This guide covers installing the Household Recovery Simulation and its dependencies.

## Requirements

- Python 3.10 or higher
- pip package manager

## Basic Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd household-recovery
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following core dependencies:

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| scipy | Statistical functions |
| networkx | Network graph creation |
| matplotlib | Visualization |
| pyyaml | Configuration files |
| tqdm | Progress bars |

## RAG Pipeline Dependencies

To use the RAG (Retrieval-Augmented Generation) features for extracting heuristics from research papers, install additional dependencies:

```bash
pip install langchain-groq google-search-results pypdf python-dotenv
```

| Package | Purpose |
|---------|---------|
| langchain-groq | LLM integration (Groq API) |
| google-search-results | SerpAPI for Google Scholar |
| pypdf | Local PDF processing |
| python-dotenv | Environment variable management |

## API Keys Setup

### For Google Scholar Integration (SerpAPI)

1. Sign up at [serpapi.com](https://serpapi.com)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export SERPAPI_KEY=your_serpapi_key_here
```

### For LLM-Based Extraction (Groq)

1. Sign up at [groq.com](https://groq.com)
2. Generate an API key
3. Set the environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### Using a .env File

Create a `.env` file in the project root:

```bash
# .env
SERPAPI_KEY=your_serpapi_key_here
GROQ_API_KEY=your_groq_api_key_here
```

The package will automatically load these variables on import.

## Development Installation

For development with testing tools:

```bash
pip install pytest pytest-cov
```

## Verify Installation

Run the simulation with default settings:

```bash
python -m household_recovery --households 10 --steps 5
```

Expected output:
```
Step 0: avg_recovery = 0.000
Step 1: avg_recovery = 0.xxx
...
Final recovery: 0.xxx
```

## Troubleshooting

### ImportError: No module named 'household_recovery'

Make sure you're running from the project root directory:
```bash
cd /path/to/household-recovery
python -m household_recovery
```

### API Key Warnings

If you see warnings about missing API keys:
```
WARNING: SERPAPI_KEY not set - will use fallback heuristics
```

This is expected if you haven't set up API keys. The simulation will use built-in fallback heuristics instead of extracting from research papers.

### Network Graph Issues

If networkx graph operations fail, ensure you have a compatible version:
```bash
pip install "networkx>=3.0"
```

## Next Steps

- [Quickstart Guide](quickstart.md) - Run your first simulation
- [Configuration Reference](configuration.md) - Customize parameters
