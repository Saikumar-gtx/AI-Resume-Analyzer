"""Deleting reports and purging old data."""

from __future__ import annotations

import logging
from datetime import timedelta

from models import Candidate, Evaluation, Job, db, utcnow

logger = logging.getLogger(__name__)


def delete_evaluation(evaluation: Evaluation) -> None:
    """Delete a report plus its candidate / pasted job when nothing else uses them."""
    candidate, job = evaluation.candidate, evaluation.job
    db.session.delete(evaluation)
    db.session.flush()
    if candidate is not None and not Evaluation.query.filter_by(candidate_id=candidate.id).count():
        db.session.delete(candidate)
    if job is not None and job.is_custom and not Evaluation.query.filter_by(job_id=job.id).count():
        db.session.delete(job)
    db.session.commit()


def purge_orphans(max_age_hours: int = 24) -> int:
    """
    Delete resumes and pasted jobs no report uses, once they are older than
    ``max_age_hours`` (e.g. uploads for the best-fit ranking that never became
    a report). Returns the number of resumes removed.
    """
    cutoff = utcnow() - timedelta(hours=max_age_hours)
    used_candidates = db.session.query(Evaluation.candidate_id)
    removed = Candidate.query.filter(
        ~Candidate.id.in_(used_candidates), Candidate.created_at < cutoff
    ).delete(synchronize_session=False)
    used_jobs = db.session.query(Evaluation.job_id).filter(Evaluation.job_id.isnot(None))
    Job.query.filter(Job.is_custom.is_(True), ~Job.id.in_(used_jobs)).delete(synchronize_session=False)
    db.session.commit()
    if removed:
        logger.info("Removed %d unused uploaded resume(s).", removed)
    return removed


def purge_older_than(days: int) -> int:
    """Delete reports older than ``days`` days. Returns how many were removed."""
    if days <= 0:
        return 0
    cutoff = utcnow() - timedelta(days=days)
    old = Evaluation.query.filter(Evaluation.created_at < cutoff).all()
    for evaluation in old:
        delete_evaluation(evaluation)
    purge_orphans(max_age_hours=0)
    if old:
        logger.info("Retention: deleted %d report(s) older than %d days.", len(old), days)
    return len(old)
