"""Keep the Job table in sync with the seed catalogue in data/jobs.json."""

from __future__ import annotations

import json
import logging

from models import Job, db
from services.scoring import skills_from_text

logger = logging.getLogger(__name__)


def load_seed_jobs(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.warning("Job seed file %s not found.", path)
        return []
    except ValueError as exc:
        logger.error("Job seed file %s is not valid JSON: %s", path, exc)
        return []

    jobs = []
    for entry in data:
        title = str(entry.get("title", "")).strip()
        description = " ".join(str(entry.get("description", "")).split())
        skills = entry.get("skills", [])
        if isinstance(skills, list):
            skills = ";".join(str(s).strip() for s in skills if str(s).strip())
        category = str(entry.get("category", "")).strip()
        if title and description:
            jobs.append({"title": title, "category": category, "description": description, "skills": skills})
    return jobs


def sync_jobs(path: str) -> tuple[int, int]:
    """
    Insert new seed jobs and update changed ones (matched by title).

    Jobs that exist only in the database are left untouched, so evaluations that
    reference them stay valid. Returns ``(added, updated)``.
    """
    seeds = load_seed_jobs(path)
    existing = {job.title: job for job in Job.query.filter_by(is_custom=False).all()}
    added = updated = 0
    for seed in seeds:
        job = existing.get(seed["title"])
        if job is None:
            db.session.add(Job(**seed))
            added += 1
        elif any(getattr(job, key) != value for key, value in seed.items()):
            for key, value in seed.items():
                setattr(job, key, value)
            updated += 1
    if added or updated:
        db.session.commit()
        logger.info("Job catalogue synced: %d added, %d updated.", added, updated)
    return added, updated


def catalogue_jobs():
    """Seeded roles only (pasted job descriptions are excluded)."""
    return Job.query.filter_by(is_custom=False)


def skill_vocabulary() -> list[str]:
    """Every skill entry across the catalogue, used to tag pasted job descriptions."""
    entries: list[str] = []
    for (skills,) in db.session.query(Job.skills).filter_by(is_custom=False):
        entries.extend(e for e in (skills or "").split(";") if e.strip())
    return list(dict.fromkeys(entries))


def create_custom_job(title: str, description: str) -> Job:
    """Store a pasted job description as a hidden Job with auto-detected skills."""
    description = " ".join(description.split())
    job = Job(
        title=title or "Custom role",
        category="Custom",
        description=description,
        skills=skills_from_text(f"{title}. {description}", skill_vocabulary()),
        is_custom=True,
    )
    db.session.add(job)
    db.session.flush()
    return job
