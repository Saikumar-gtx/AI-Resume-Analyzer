"""Score a resume against one job, or rank it against many."""

from __future__ import annotations

from services.embeddings import Embedder
from services.scoring import ScoreBreakdown, score_resume


def job_text(job) -> str:
    return f"{job.title}. {job.description}"


def analyze_resume(resume_text: str, job, embedder: Embedder | None) -> ScoreBreakdown:
    similarity = None
    if embedder is not None and embedder.enabled:
        similarity = embedder.resume_job_similarity(resume_text, job_text(job))
    return score_resume(resume_text, job.description, job.skills, similarity)


def rank_roles(resume_text: str, jobs: list, embedder: Embedder | None) -> list[tuple[object, ScoreBreakdown]]:
    """Every job scored against the resume, best fit first."""
    sims = None
    if embedder is not None and embedder.enabled:
        sims = embedder.resume_job_similarities(resume_text, [job_text(j) for j in jobs])
    ranked = [
        (job, score_resume(resume_text, job.description, job.skills, sims[i] if sims else None))
        for i, job in enumerate(jobs)
    ]
    ranked.sort(key=lambda pair: (-pair[1].overall, pair[0].title))
    return ranked
