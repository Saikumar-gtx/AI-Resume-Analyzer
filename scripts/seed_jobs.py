"""
Sync the Job table with data/jobs.json (adds new roles, updates changed ones).

Usage:  python -m scripts.seed_jobs
"""

from app import create_app
from models import Job
from services.jobs import sync_jobs


def main() -> None:
    app = create_app({"RAG_WARMUP": False})
    with app.app_context():
        added, updated = sync_jobs(app.config["JOBS_SEED_FILE"])
        print(f"{added} added, {updated} updated, {Job.query.count()} jobs total.")


if __name__ == "__main__":
    main()
