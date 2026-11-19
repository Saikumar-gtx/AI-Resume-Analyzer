"""SQLAlchemy models."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

db = SQLAlchemy()


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone support)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))
    skills_text = db.Column(db.Text)
    resume_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utcnow)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(60), default="")
    description = db.Column(db.Text, nullable=False)
    # Format documented in services.scoring.parse_skills.
    skills = db.Column(db.Text, default="")
    # True for job descriptions pasted by a user (hidden from the role catalogue).
    is_custom = db.Column(db.Boolean, default=False, nullable=False)

    @property
    def skill_names(self) -> list[str]:
        from services.scoring import parse_skills

        return [s.name for s in parse_skills(self.skills)]


STATUS_PENDING = "pending"
STATUS_DONE = "done"


class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidate.id"), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id"), nullable=True)
    score = db.Column(db.Float, default=0.0)
    summary = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    # JSON blob: score breakdown, matched/missing skills, strengths, gaps, source.
    details = db.Column(db.Text)
    # "pending" while feedback is being written in the background, then "done".
    status = db.Column(db.String(20), default=STATUS_DONE, nullable=False)
    # Hash of resume text + job, used to reuse results for identical submissions.
    cache_key = db.Column(db.String(64), index=True)
    created_at = db.Column(db.DateTime, default=utcnow)

    candidate = db.relationship("Candidate", backref="evaluations")
    job = db.relationship("Job", backref="evaluations")

    @property
    def details_dict(self) -> dict:
        if not self.details:
            return {}
        try:
            data = json.loads(self.details)
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @property
    def is_pending(self) -> bool:
        return self.status == STATUS_PENDING

    @property
    def recommendation_list(self) -> list[str]:
        items = []
        for line in (self.recommendations or "").splitlines():
            line = line.strip().lstrip("•-* ").strip()
            if line:
                items.append(line)
        return items


# Columns added after the first release. `db.create_all()` never alters existing
# tables, so add them in place for databases created by older versions.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "evaluation": {
        "details": "TEXT",
        "status": "VARCHAR(20) NOT NULL DEFAULT 'done'",
        "cache_key": "VARCHAR(64)",
    },
    "job": {
        "category": "VARCHAR(60) DEFAULT ''",
        "is_custom": "BOOLEAN NOT NULL DEFAULT 0",
    },
}


def upgrade_schema() -> None:
    """Add any columns missing from tables created by an older version."""
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    with db.engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            if table not in tables:
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))
