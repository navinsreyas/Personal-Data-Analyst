# Personal Data Analyst

Ask questions about your CSV data in plain English.
The system classifies your question, builds an analysis plan,
writes and executes Python code, and returns a formatted answer — all automatically.

Built with **LangGraph** · **smolagents** · **Claude Sonnet** · **Streamlit** · **pandas**

---

## Architecture
```
Your question
    │
    ▼
[Router]         — Is this answerable? (green / yellow / red)
    │
    ▼
[Planner]        — Creates a structured analysis plan
    │
    ▼
[Reviewer]       — Validates the plan before execution
    │
    ▼
[Supervisor]     — Intelligent routing: execute / re-plan / refuse
    │
    ▼
[CodeAgent]      — Writes + runs Python using 5 data tools (smolagents)
    │
    ▼
[Explainer]      — Formats the result into a clear markdown answer
```

---


## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key in .env
ANTHROPIC_API_KEY=your-key-here

# 3. Scan your CSV to generate schema profile
python scanner.py data/olist_sample.csv --output data/schema_profile.json --grain "1 row = 1 order"

# 4. Launch the Streamlit app
python -m streamlit run app.py
```

---

## API

A FastAPI wrapper is included for programmatic access.

**Run locally:**
```bash
pip install fastapi uvicorn
python -m uvicorn api.main:app --port 8000
```

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Natural language → answer |
| `/health` | GET | Liveness check |
| `/schema` | GET | Dataset schema info |
| `/docs` | GET | Interactive Swagger UI |

**Example:**
```bash
curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"question": "What are the top 5 product categories by revenue?"}'
```

---

## Project Structure
```
├── agent/
│   ├── graph_builder.py   — LangGraph state machine assembly
│   ├── nodes.py           — All node implementations (router, planner, etc.)
│   ├── prompts.py         — All LLM system prompts
│   ├── state.py           — AgentState TypedDict definition
│   ├── tools.py           — smolagents @tool functions (load, filter, aggregate)
│   ├── models.py          — Pydantic output models
│   ├── memory.py          — Semantic plan cache
│   └── _shared_model.py   — Custom Anthropic model wrapper for smolagents
│
├── data/
│   └── schema_profile.json — Pre-computed dataset profile (from scanner.py)
│
├── app.py                 — Streamlit chat UI
├── scanner.py             — CSV profiling tool (generates schema_profile.json)
├── eval.py                — Evaluation runner (20 ground truth test cases)
├── eval_cases.py          — Ground truth answers
└── olist_sample.csv       — Sample dataset (Brazilian e-commerce, 10k orders)
```

---

## The 5 Data Tools

The `smolagents` CodeAgent has access to exactly 5 tools:

| Tool | What it does |
|---|---|
| `load_csv_data(path)` | Loads the CSV, auto-parses datetimes, returns JSON |
| `calculate_statistics(data, col)` | Returns mean/std/min/max/percentiles |
| `group_and_aggregate(data, group, col, fn)` | GroupBy + agg (sum/mean/count/min/max) |
| `filter_rows(data, col, op, val)` | Filters rows by numeric condition |
| `generate_chart(data, type, x, y, title)` | Generates bar/line/histogram/pie charts |

---

## Evaluation Results

Tested against 20 ground truth questions across three difficulty levels:

| Difficulty | Score | Example Question |
|------------|-------|-----------------|
| Simple | 5/5 (100%) | "How many unique customer states?" |
| Medium | 5/5 (100%) | "What % of orders were delivered?" |
| Complex | 9/10 (90%) | "Which state has fastest delivery?" |
| **Overall** | **19/20 (95%)** | |

*One failure (C09) was a checker formatting issue — the pipeline answered correctly but formatted "November 2017" instead of "2017-11".*

Run the eval suite:

```bash
python eval.py                      # all 20 cases
python eval.py --difficulty simple  # 5 simple cases only
python eval.py --id C03             # single case
python eval.py --dry-run            # preview without API calls
```

---

## LangSmith Tracing

Add your LangSmith API key to `.env` to enable full trace visibility:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=personal-data-analyst
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```
