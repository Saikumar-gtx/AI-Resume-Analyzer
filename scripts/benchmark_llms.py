"""
Compare the configured LLMs on the same resume/role: latency, valid output,
and how many statements the grounding check had to remove.

Usage:
    python -m scripts.benchmark_llms RESUME_FILE [--role "AI Engineer"] [--runs 1]
                                     [--models groq:openai/gpt-oss-120b ollama:mistral:7b ...]

Results are printed as a table and saved to instance/benchmarks/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from config import Config
from services.advisor import SYSTEM_PROMPT, build_prompt, ground_feedback, parse_feedback
from services.embeddings import Embedder
from services.llm import GroqClient, LLMError, OllamaClient
from services.rag import KnowledgeBase
from services.resume_parser import extract_text, file_extension
from services.scoring import score_resume


def make_client(spec: str):
    provider, _, model = spec.partition(":")
    if provider == "groq":
        return GroqClient(Config.GROQ_API_KEY, [model], Config.GROQ_BASE_URL, timeout=Config.GROQ_TIMEOUT)
    if provider == "ollama":
        return OllamaClient(Config.OLLAMA_BASE_URL, model, timeout=Config.OLLAMA_TIMEOUT)
    raise SystemExit(f"Unknown provider in {spec!r}; use groq:<model> or ollama:<model>")


def main() -> None:
    default_models = [f"groq:{m}" for m in Config.GROQ_MODELS] + [f"ollama:{Config.OLLAMA_MODEL}"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resume")
    parser.add_argument("--role", default="AI Engineer", help="title from data/jobs.json")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=default_models)
    args = parser.parse_args()

    with open(args.resume, "rb") as fh:
        resume = extract_text(fh, file_extension(args.resume))
    with open(Config.JOBS_SEED_FILE, encoding="utf-8") as fh:
        jobs = {j["title"]: j for j in json.load(fh)}
    job = jobs[args.role]
    skills = ";".join(job["skills"])

    embedder = Embedder(Config.EMBED_MODEL_NAME, enabled=Config.ENABLE_EMBEDDINGS)
    knowledge = KnowledgeBase(embedder, Config.KNOWLEDGE_FILE, Config.EMBED_CACHE_DIR)
    breakdown = score_resume(
        resume, job["description"], skills, embedder.resume_job_similarity(resume, job["description"])
    )
    context = knowledge.retrieve(f"Advice for a {args.role} resume. Missing: {', '.join(breakdown.missing_skills)}.")
    prompt = build_prompt(resume, args.role, job["description"], breakdown, context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    print(f"Role: {args.role} | score {breakdown.overall} | {len(context)} RAG chunks | {args.runs} run(s)\n")

    results = []
    for spec in args.models:
        client = make_client(spec)
        runs = []
        for i in range(args.runs):
            started = time.time()
            try:
                raw, _ = client.complete(messages)
                elapsed = time.time() - started
                fb = parse_feedback(raw)
                removed = ground_feedback(fb, resume, breakdown, skills).grounding_removed if fb else None
                runs.append({"ok": True, "seconds": elapsed, "parsed": fb is not None, "removed": removed})
            except LLMError as exc:
                runs.append({"ok": False, "seconds": time.time() - started, "error": str(exc)})
            print(f"  {spec} run {i + 1}: {runs[-1]}")
        ok = [r for r in runs if r["ok"]]
        results.append(
            {
                "model": spec,
                "success_rate": len(ok) / len(runs),
                "parse_rate": sum(r["parsed"] for r in ok) / len(runs),
                "median_seconds": round(statistics.median(r["seconds"] for r in ok), 1) if ok else None,
                "avg_grounding_removed": (
                    round(statistics.mean(r["removed"] for r in ok if r["removed"] is not None), 2)
                    if any(r.get("removed") is not None for r in ok) else None
                ),
                "runs": runs,
            }
        )

    print(f"\n{'model':<32} {'success':>8} {'parsed':>7} {'median s':>9} {'removed':>8}")
    for r in results:
        print(
            f"{r['model']:<32} {r['success_rate']:>8.0%} {r['parse_rate']:>7.0%} "
            f"{r['median_seconds'] if r['median_seconds'] is not None else '-':>9} "
            f"{r['avg_grounding_removed'] if r['avg_grounding_removed'] is not None else '-':>8}"
        )

    out_dir = Path(Config.EMBED_CACHE_DIR) / "benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"role": args.role, "score": breakdown.overall, "results": results}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
