"""
Create tables, sync the job catalogue and pre-build the RAG embedding cache.

Usage:  python -m scripts.init_db [--reset]
"""

import argparse

from app import create_app
from models import Job, db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop all tables first (deletes all evaluations)")
    args = parser.parse_args()

    app = create_app({"RAG_WARMUP": False})
    with app.app_context():
        if args.reset:
            db.drop_all()
            print("Dropped all tables.")
            db.create_all()
            from services.jobs import sync_jobs

            sync_jobs(app.config["JOBS_SEED_FILE"])
        print(f"Database ready with {Job.query.count()} jobs.")

    knowledge = app.extensions["analyzer"]["knowledge"]
    if knowledge.ensure_ready():
        print(f"RAG index ready ({len(knowledge.documents)} chunks) - cache at {knowledge.cache_path}")
    else:
        print("RAG index unavailable (embedding model could not be loaded).")


if __name__ == "__main__":
    main()
