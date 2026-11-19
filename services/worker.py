"""Write AI feedback in the background so slow models never block a request."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor

from models import STATUS_DONE, Evaluation, db
from services.advisor import Feedback, generate_feedback, rule_based_feedback
from services.scoring import ScoreBreakdown

logger = logging.getLogger(__name__)


def cache_key(resume_text: str, job_title: str, job_description: str, job_skills: str | None) -> str:
    """Stable hash of everything that determines a report's content."""
    normalized = " ".join(resume_text.lower().split())
    payload = "\x1f".join([normalized, job_title, " ".join(job_description.split()), job_skills or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def breakdown_from(evaluation: Evaluation) -> ScoreBreakdown:
    data = evaluation.details_dict.get("breakdown", {})
    fields = ScoreBreakdown.__dataclass_fields__
    return ScoreBreakdown(**{k: v for k, v in data.items() if k in fields})


def apply_feedback(evaluation: Evaluation, feedback: Feedback, breakdown: ScoreBreakdown) -> None:
    evaluation.summary = feedback.summary
    evaluation.recommendations = "\n".join(f"• {r}" for r in feedback.recommendations)
    evaluation.details = json.dumps(
        {
            "breakdown": breakdown.to_dict(),
            "strengths": feedback.strengths,
            "gaps": feedback.gaps,
            "source": feedback.source,
            "model": feedback.model,
            "grounding_removed": feedback.grounding_removed,
        }
    )
    evaluation.status = STATUS_DONE


class FeedbackWorker:
    """Runs ``generate_feedback`` for pending evaluations on a small thread pool."""

    def __init__(self, app, services: dict, max_workers: int = 2, inline: bool = False):
        self.app = app
        self.services = services
        self.inline = inline
        self._pool = None if inline else ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="feedback")

    def submit(self, evaluation_id: int) -> Future | None:
        if self.inline:
            self._run(evaluation_id)
            return None
        return self._pool.submit(self._run, evaluation_id)

    def _run(self, evaluation_id: int) -> None:
        with self.app.app_context():
            evaluation = db.session.get(Evaluation, evaluation_id)
            if evaluation is None or not evaluation.is_pending:
                return
            breakdown = breakdown_from(evaluation)
            job = evaluation.job
            resume_text = evaluation.candidate.resume_text or ""
            started = time.time()
            try:
                feedback = generate_feedback(
                    resume_text,
                    job.title,
                    job.description,
                    breakdown,
                    self.services["knowledge"],
                    self.services["llm"],
                    top_k=self.app.config["RAG_TOP_K"],
                    job_skills=job.skills,
                    vocabulary=self.services["vocabulary"](),
                )
            except Exception:  # generate_feedback shouldn't raise; never leave a report stuck
                logger.exception("Feedback generation crashed for evaluation %d", evaluation_id)
                feedback = rule_based_feedback(resume_text, job.title, breakdown)

            # The report may have been deleted while the model was thinking.
            evaluation = db.session.get(Evaluation, evaluation_id)
            if evaluation is None:
                return
            apply_feedback(evaluation, feedback, breakdown)
            db.session.commit()
            logger.info(
                "Evaluation %d feedback ready (%s, %s) in %.1fs",
                evaluation_id, feedback.source, feedback.model or "-", time.time() - started,
            )

    def finish_stale(self, evaluation: Evaluation) -> None:
        """Fallback for a report whose background job was lost (e.g. server restart)."""
        breakdown = breakdown_from(evaluation)
        feedback = rule_based_feedback(evaluation.candidate.resume_text or "", evaluation.job.title, breakdown)
        apply_feedback(evaluation, feedback, breakdown)
        db.session.commit()


class RateLimiter:
    """In-memory sliding-window limiter keyed by client (per process)."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
