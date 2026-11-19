# AI Resume Analyzer

A Flask web app that scores a resume against a target role and returns actionable feedback. It combines skill matching, keyword overlap and semantic similarity into a 0–100 score. It then uses retrieval-augmented generation (RAG) with a hosted LLM (Groq), falling back to a local one (Ollama), to write a summary, strengths, gaps and five concrete edits. Every AI claim is fact-checked against the resume before it is shown.

The app also works **without an API key or ML model**. It falls back to rule-based feedback and keyword-only scoring.

## Features

- **Any target role**: pick one of 35 curated roles, or **paste any job posting**. Its skills are detected automatically from a vocabulary of several hundred tools and technologies.
- **Best-fit roles**: upload once on the Roles page and get your resume ranked against all 35 roles, with missing skills per role. It's instant with no AI call, and any row opens a full report without re-uploading.
- **PDF, DOCX and TXT uploads**, parsed in memory. DOCX tables are included and scanned/empty PDFs are detected.
- **Section-aware, explainable score**:
  | Component | Weight | What it measures |
  |---|---|---|
  | Skill coverage | 50% | Share of the role's skills in the resume. A skill **used** in experience/projects counts fully; one only in the skills list ("listed only") counts 60%; one **described but not named** ("Helm charts on EKS" for Kubernetes) counts 50% |
  | Keyword overlap | 20% | Share of the job description's most frequent meaningful terms found |
  | Semantic similarity | 30% | Embedding similarity (`all-MiniLM-L6-v2`) between resume chunks and the job description |

  If a component is unavailable, the remaining weights are rescaled. "Described, not named" evidence comes from a curated phrase list (`services/concepts.py`). Embedding similarity was tested for this and rejected: with both MiniLM and bge-m3, true and false skill/bullet pairs overlapped, so it would have handed out false credit.
- **Instant score, background feedback**: the score appears right away while the AI review is written in the background. The report page updates by itself, so slow local models never block or time out a request.
- **RAG feedback**: relevant tips are retrieved from `knowledge/resume_guidelines.txt` with FAISS and passed to the LLM with the resume, the role and the found/missing/listed-only skills.
- **Fallback chain**: Groq primary model → Groq fallback model → **local Ollama** (Mistral 7B by default) → rule-based checks. Each report shows which model wrote it.
- **Fact check (grounding)**: AI statements that contradict the resume are removed automatically, such as "pytest is not mentioned" when it is, praise for a skill the resume never shows, or advice to list a skill the candidate hasn't demonstrated.
- **Progress tracking**: re-analyze a revised resume for the same role and the report shows the score change since last time.
- **Privacy controls**: delete any report along with its stored resume text, set an automatic retention period, or hide `/history` on public deployments.
- **Abuse and cost protection**: per-client rate limit, and identical submissions (same resume + same role) reuse the saved AI report instead of calling the model again.
- Editorial-style UI with five tabs: **Home**, **Analyze**, **Roles**, **Report**, **History**. Responsive, print-to-PDF styling, CSRF protection, upload limits and a `/health` endpoint.

## Project structure

```
app.py                 Flask app factory and routes
config.py              Settings, read from environment / .env
models.py              SQLAlchemy models (+ in-place column upgrades)
services/
  resume_parser.py     PDF/DOCX/TXT text extraction, contact details
  sections.py          Detect resume sections (experience, projects, skills, ...)
  concepts.py          Phrases that show a skill without naming it
  analysis.py          Score one role, or rank all roles in one pass
  scoring.py           Skill/keyword matching, evidence weighting, pasted-job skills
  embeddings.py        Lazy SentenceTransformer wrapper (local cache first)
  rag.py               Knowledge base + FAISS index with on-disk cache
  llm.py               Groq + Ollama clients and the provider fallback chain
  advisor.py           Prompt, parsing, grounding check, rule-based feedback
  worker.py            Background feedback jobs, result cache key, rate limiter
  jobs.py              Role catalogue sync and pasted job postings
  maintenance.py       Report deletion and retention
  text.py              Text cleaning and chunking
data/jobs.json         Role catalogue (title, category, description, skills)
knowledge/             RAG source documents
templates/, static/    UI
scripts/               init_db.py, seed_jobs.py, benchmark_llms.py
tests/                 pytest suite (no network, no ML model needed)
.github/workflows/     CI: ruff + pytest on Python 3.11 and 3.13
```

## Getting started

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY and FLASK_SECRET_KEY
python app.py
```

Open http://127.0.0.1:5000. On first start the database is created in `instance/` and the role catalogue is seeded. The embedding model (~90 MB) downloads in the background once, and later starts load it from the local cache.

For the offline fallback, install [Ollama](https://ollama.com) and run `ollama pull mistral:7b`.

Set `FLASK_DEBUG=1` to enable the debugger and auto-reload.

### Configuration

All settings are optional; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Enables hosted AI feedback |
| `GROQ_PRIMARY_MODEL` / `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Groq models tried in order |
| `OLLAMA_ENABLED` | `true` | Use a local Ollama model when Groq fails or has no key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | From Docker use `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` / `OLLAMA_TIMEOUT` | `mistral:7b` / `420` | Model must be pulled first: `ollama pull mistral:7b` |
| `FLASK_SECRET_KEY` | dev key | **Set this in production** |
| `DATABASE_URL` | `sqlite:///instance/resume_analyzer.db` | Any SQLAlchemy URL |
| `MAX_UPLOAD_MB` | `5` | Upload size limit |
| `ENABLE_EMBEDDINGS` | `true` | Set `false` to skip the ML model (keyword-only scoring, no RAG) |
| `ENABLE_HISTORY` | `true` | Set `false` on public deployments to hide `/history`; visitors can still see their own reports |
| `RATE_LIMIT_PER_10_MIN` | `10` | Analyses per client per 10 minutes (`0` disables) |
| `CACHE_DAYS` | `7` | Reuse an identical AI report for this long (`0` disables) |
| `RETENTION_DAYS` | `0` | Delete reports older than this at startup (`0` keeps forever) |
| `ASYNC_FEEDBACK` / `FEEDBACK_WORKERS` | `true` / `2` | Write feedback in background threads |

### Managing roles

Edit `data/jobs.json` and restart (or run `python -m scripts.seed_jobs`). Roles are matched by title: new ones are added and changed ones are updated. `category` is one of `Software`, `Data & AI`, `Infrastructure` or `Product & More`. Skills use `|` for aliases and an optional `Label=` prefix for umbrella entries:

```json
{
  "title": "Data Scientist",
  "category": "Data & AI",
  "description": "Analyze large datasets…",
  "skills": ["Python", "scikit-learn|sklearn", "BI tool=Tableau|Power BI|Looker"]
}
```

Skills in the catalogue also form the vocabulary used to tag pasted job postings, so adding roles makes pasted-posting detection better too.

### Database

```bash
python -m scripts.init_db            # create tables, seed roles, pre-build the RAG cache
python -m scripts.init_db --reset    # drop everything and start fresh
```

## Comparing models

```bash
python -m scripts.benchmark_llms path/to/resume.pdf --role "AI Engineer" --runs 3
```

This runs the same prompt through every configured model (Groq primary, Groq fallback, Ollama) and reports success rate, valid-output rate, median latency and how many statements the fact check removed. Results are saved to `instance/benchmarks/`.

## Tests and linting

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest
```

The tests run without network access, a real LLM or the embedding model. CI runs both on every push.

## Deployment

**Docker** (CPU-only PyTorch, embedding model baked into the image, served with gunicorn):

```bash
docker compose up --build
```

**Heroku-style platforms** use the `Procfile`: `gunicorn "app:create_app()"`.

Background jobs run in threads inside each gunicorn worker. If a worker restarts mid-job, the report is finished with rule-based feedback the next time it is opened.
