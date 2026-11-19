"""
AI Resume Analyzer — Flask application.

Run locally with ``python app.py`` or ``flask --app app run``; in production use
``gunicorn "app:create_app()"``.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
from datetime import timedelta

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from models import STATUS_DONE, STATUS_PENDING, Candidate, Evaluation, Job, db, upgrade_schema, utcnow
from services.analysis import analyze_resume, rank_roles
from services.embeddings import Embedder
from services.jobs import catalogue_jobs, create_custom_job, skill_vocabulary, sync_jobs
from services.llm import GroqClient, LLMChain, OllamaClient
from services.maintenance import delete_evaluation, purge_older_than, purge_orphans
from services.rag import KnowledgeBase
from services.resume_parser import ResumeParseError, extract_contact, extract_text, file_extension
from services.scoring import score_band
from services.worker import FeedbackWorker, RateLimiter, cache_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
# Hugging Face logs every model-file check at INFO; keep the console readable.
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    if Config.uses_dev_secret(app.config["SECRET_KEY"]) and not app.debug and not app.testing:
        logger.warning("FLASK_SECRET_KEY is not set; using an insecure development key.")

    os.makedirs(app.instance_path, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        db.create_all()
        upgrade_schema()
        sync_jobs(app.config["JOBS_SEED_FILE"])
        purge_older_than(app.config["RETENTION_DAYS"])
        purge_orphans()
        logger.info("%d roles available.", catalogue_jobs().count())

    embedder = Embedder(app.config["EMBED_MODEL_NAME"], enabled=app.config["ENABLE_EMBEDDINGS"])
    knowledge = KnowledgeBase(embedder, app.config["KNOWLEDGE_FILE"], app.config["EMBED_CACHE_DIR"])
    groq = GroqClient(
        app.config["GROQ_API_KEY"],
        app.config["GROQ_MODELS"],
        app.config["GROQ_BASE_URL"],
        timeout=app.config["GROQ_TIMEOUT"],
    )
    ollama = OllamaClient(
        app.config["OLLAMA_BASE_URL"],
        app.config["OLLAMA_MODEL"],
        timeout=app.config["OLLAMA_TIMEOUT"],
        enabled=app.config["OLLAMA_ENABLED"],
    )
    # Groq primary -> Groq fallback -> local Ollama -> (advisor) rule-based feedback.
    llm = LLMChain([groq, ollama])
    services = {
        "embedder": embedder,
        "knowledge": knowledge,
        "llm": llm,
        "groq": groq,
        "ollama": ollama,
        "vocabulary": skill_vocabulary,
        "limiter": RateLimiter(app.config["RATE_LIMIT_PER_10_MIN"], 600),
    }
    services["worker"] = FeedbackWorker(
        app, services, max_workers=app.config["FEEDBACK_WORKERS"], inline=not app.config["ASYNC_FEEDBACK"]
    )
    app.extensions["analyzer"] = services

    if not groq.available:
        logger.warning("GROQ_API_KEY is not set; AI feedback will use Ollama if it is running.")
    if not llm.available:
        logger.warning("No LLM configured; using rule-based feedback only.")
    if app.config["RAG_WARMUP"] and app.config["ENABLE_EMBEDDINGS"]:
        # Load the embedding model and RAG index in the background so startup stays
        # fast but the first analysis doesn't pay the ~20 s model load.
        def warmup() -> None:
            knowledge.ensure_ready()
            embedder.encode(["warmup"])

        threading.Thread(target=warmup, name="rag-warmup", daemon=True).start()

    _register_template_helpers(app)
    _register_routes(app)
    _register_error_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Template helpers & CSRF
# ---------------------------------------------------------------------------

def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = session["_csrf_token"] = secrets.token_urlsafe(32)
    return token


def _csrf_valid() -> bool:
    expected = session.get("_csrf_token")
    supplied = request.form.get("csrf_token", "")
    return bool(expected) and hmac.compare_digest(expected, supplied)


# Display order and glyphs for role categories (see data/jobs.json).
CATEGORIES = [
    ("Software", "{ }"),
    ("Data & AI", "∑"),
    ("Infrastructure", "#"),
    ("Product & More", "◇"),
]


def _register_template_helpers(app: Flask) -> None:
    @app.context_processor
    def inject_globals():
        # The session may reference reports that were deleted (e.g. after a DB reset).
        latest = next(
            (rid for rid in session.get("my_reports", [])[:5] if db.session.get(Evaluation, rid)),
            None,
        )
        return {
            "csrf_token": _csrf_token,
            "history_enabled": app.config["ENABLE_HISTORY"],
            "score_band": score_band,
            "latest_report_id": latest,
            "today": utcnow(),
            "categories": CATEGORIES,
        }

    @app.template_filter("datetime")
    def format_datetime(value):
        return value.strftime("%d %b %Y, %H:%M UTC") if value else ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _register_routes(app: Flask) -> None:
    services = app.extensions["analyzer"]

    @app.get("/")
    def index():
        return render_template(
            "home.html",
            job_count=catalogue_jobs().count(),
            report_count=Evaluation.query.count(),
            ai_enabled=services["llm"].available,
        )

    @app.get("/roles")
    def roles():
        jobs = catalogue_jobs().order_by(Job.title.asc()).all()
        return render_template("roles.html", jobs=jobs)

    @app.get("/analyze")
    def analyze_form():
        jobs = catalogue_jobs().order_by(Job.title.asc()).all()
        jobs_data = {
            str(j.id): {"description": j.description, "skills": j.skill_names, "category": j.category or ""}
            for j in jobs
        }
        selected = request.args.get("job", type=int)
        return render_template(
            "analyze.html",
            jobs=jobs,
            jobs_data=jobs_data,
            selected_job=selected if str(selected) in jobs_data else None,
            mode="custom" if request.args.get("mode") == "custom" else "catalogue",
            max_upload_mb=app.config["MAX_UPLOAD_MB"],
            min_jd_chars=app.config["MIN_JOB_DESCRIPTION_CHARS"],
            ai_enabled=services["llm"].available,
        )

    @app.post("/analyze")
    def analyze():
        if app.config["CSRF_ENABLED"] and not _csrf_valid():
            flash("Your session expired. Please try again.", "error")
            return redirect(url_for("analyze_form"))

        mode = "custom" if request.form.get("mode") == "custom" else "catalogue"
        job_id = request.form.get("job_id")
        retry = redirect(url_for("analyze_form", mode=mode, job=job_id if mode == "catalogue" else None))

        # --- target job ---
        custom_title = " ".join(request.form.get("custom_title", "").split())[:120]
        custom_description = request.form.get("custom_description", "").strip()[:20000]
        if mode == "custom":
            if len(custom_description) < app.config["MIN_JOB_DESCRIPTION_CHARS"]:
                flash("Paste the full job description (at least a few sentences).", "error")
                return retry
            job = None
        else:
            job = _get_job(job_id)
            if job is None or job.is_custom:
                flash("Please select a target job.", "error")
                return redirect(url_for("analyze_form"))

        # --- resume ---
        resume_text, error = _read_resume_upload(app)
        if error:
            flash(error, "error")
            return retry
        if not services["limiter"].allow(request.remote_addr or "unknown"):
            flash("You have run a lot of analyses in a short time. Please wait a few minutes.", "error")
            return retry

        if job is None:
            job = create_custom_job(custom_title, custom_description)
        candidate = _new_candidate(resume_text, request.form.get("candidate_name", ""))
        evaluation_id = _start_evaluation(app, services, candidate, job)
        # Post/Redirect/Get so refreshing the report doesn't re-submit the upload.
        return redirect(url_for("report", evaluation_id=evaluation_id))

    # --- Best-fit roles: upload once, rank against the whole catalogue ---------

    @app.post("/match")
    def match_upload():
        if app.config["CSRF_ENABLED"] and not _csrf_valid():
            flash("Your session expired. Please try again.", "error")
            return redirect(url_for("roles"))
        resume_text, error = _read_resume_upload(app)
        if error:
            flash(error, "error")
            return redirect(url_for("roles"))
        if not services["limiter"].allow(request.remote_addr or "unknown"):
            flash("You have run a lot of analyses in a short time. Please wait a few minutes.", "error")
            return redirect(url_for("roles"))
        candidate = _new_candidate(resume_text, request.form.get("candidate_name", ""))
        db.session.add(candidate)
        db.session.commit()
        mine = [c for c in session.get("my_candidates", []) if c != candidate.id]
        session["my_candidates"] = [candidate.id, *mine][:20]
        return redirect(url_for("match_results", candidate_id=candidate.id))

    @app.get("/match/<int:candidate_id>")
    def match_results(candidate_id: int):
        candidate = _my_candidate(candidate_id)
        jobs = catalogue_jobs().order_by(Job.title.asc()).all()
        ranked = rank_roles(candidate.resume_text or "", jobs, services["embedder"])
        return render_template("match.html", candidate=candidate, ranked=ranked)

    @app.post("/match/<int:candidate_id>/report")
    def match_report(candidate_id: int):
        if app.config["CSRF_ENABLED"] and not _csrf_valid():
            flash("Your session expired. Please try again.", "error")
            return redirect(url_for("match_results", candidate_id=candidate_id))
        candidate = _my_candidate(candidate_id)
        job = _get_job(request.form.get("job_id"))
        if job is None or job.is_custom:
            abort(404)
        if not services["limiter"].allow(request.remote_addr or "unknown"):
            flash("You have run a lot of analyses in a short time. Please wait a few minutes.", "error")
            return redirect(url_for("match_results", candidate_id=candidate_id))
        return redirect(url_for("report", evaluation_id=_start_evaluation(app, services, candidate, job)))

    @app.get("/report/<int:evaluation_id>")
    def report(evaluation_id: int):
        evaluation = _visible_report(app, evaluation_id)
        if evaluation.is_pending and _is_stale(app, evaluation):
            services["worker"].finish_stale(evaluation)
        details = evaluation.details_dict
        previous = _previous_report(evaluation)
        return render_template(
            "results.html",
            evaluation=evaluation,
            breakdown=details.get("breakdown", {}),
            strengths=details.get("strengths", []),
            gaps=details.get("gaps", []),
            source=details.get("source"),  # None for reports saved by older versions
            model=details.get("model"),
            grounding_removed=details.get("grounding_removed", 0),
            score=round(evaluation.score or 0),
            previous=previous,
            delta=round((evaluation.score or 0) - (previous.score or 0)) if previous else None,
        )

    @app.get("/report/<int:evaluation_id>/status")
    def report_status(evaluation_id: int):
        evaluation = _visible_report(app, evaluation_id)
        return jsonify(status=evaluation.status, ready=evaluation.status == STATUS_DONE)

    @app.post("/report/<int:evaluation_id>/delete")
    def delete_report(evaluation_id: int):
        if app.config["CSRF_ENABLED"] and not _csrf_valid():
            flash("Your session expired. Please try again.", "error")
            return redirect(url_for("report", evaluation_id=evaluation_id))
        evaluation = _visible_report(app, evaluation_id)
        delete_evaluation(evaluation)
        session["my_reports"] = [r for r in session.get("my_reports", []) if r != evaluation_id]
        flash("Report deleted, along with the resume text it stored.", "notice")
        return redirect(url_for("history") if app.config["ENABLE_HISTORY"] else url_for("analyze_form"))

    @app.get("/history")
    def history():
        if not app.config["ENABLE_HISTORY"]:
            abort(404)
        evaluations = (
            Evaluation.query.order_by(Evaluation.created_at.desc(), Evaluation.id.desc()).limit(100).all()
        )
        return render_template("history.html", evaluations=evaluations)

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            jobs=catalogue_jobs().count(),
            ai_feedback=services["llm"].available,
            groq=services["groq"].available,
            ollama=services["ollama"].ping(),
            embeddings=services["embedder"].enabled,
        )


def _get_job(raw_id: str | None) -> Job | None:
    try:
        return db.session.get(Job, int(raw_id or ""))
    except (TypeError, ValueError):
        return None


def _read_resume_upload(app: Flask) -> tuple[str, str | None]:
    """``(resume_text, error_message)`` for the uploaded ``resume`` file."""
    upload = request.files.get("resume")
    if upload is None or not upload.filename:
        return "", "Please choose a resume file to upload."
    ext = file_extension(upload.filename)
    if ext not in app.config["ALLOWED_EXTENSIONS"]:
        return "", "Unsupported file type. Upload a PDF, DOCX or TXT resume."
    try:
        return extract_text(upload.stream, ext), None
    except ResumeParseError as exc:
        return "", str(exc)


def _new_candidate(resume_text: str, raw_name: str) -> Candidate:
    contact = extract_contact(resume_text)
    return Candidate(
        name=" ".join(raw_name.split())[:120] or "Candidate",
        email=contact["email"],
        phone=contact["phone"],
        resume_text=resume_text,
    )


def _my_candidate(candidate_id: int) -> Candidate:
    """A candidate uploaded in this browser session (resume text is private)."""
    candidate = db.session.get(Candidate, candidate_id)
    if candidate is None or candidate_id not in session.get("my_candidates", []):
        abort(404)
    return candidate


def _start_evaluation(app: Flask, services: dict, candidate: Candidate, job: Job) -> int:
    """Score now, queue the AI feedback, and return the report id (or a cached one)."""
    resume_text = candidate.resume_text or ""
    key = cache_key(resume_text, job.title, job.description, job.skills)
    cached = _cached_report(app, key)
    if cached is not None:
        if candidate.id is None:
            db.session.rollback()  # discard the unsaved candidate / pasted job
        _remember_report(cached.id)
        flash("You already analyzed this resume for this role, so here is that report.", "notice")
        return cached.id

    breakdown = analyze_resume(resume_text, job, services["embedder"])
    candidate.skills_text = ";".join(breakdown.matched_skills)
    evaluation = Evaluation(
        candidate=candidate,
        job=job,
        score=breakdown.overall,
        summary="",
        recommendations="",
        details=json.dumps({"breakdown": breakdown.to_dict()}),
        status=STATUS_PENDING,
        cache_key=key,
    )
    db.session.add_all([candidate, evaluation])
    db.session.commit()
    logger.info("Evaluation %d: %s vs %s -> %d", evaluation.id, candidate.name, job.title, breakdown.overall)
    services["worker"].submit(evaluation.id)
    _remember_report(evaluation.id)
    return evaluation.id


def _remember_report(evaluation_id: int) -> None:
    reports = [r for r in session.get("my_reports", []) if r != evaluation_id]
    session["my_reports"] = [evaluation_id, *reports][:50]


def _visible_report(app: Flask, evaluation_id: int) -> Evaluation:
    evaluation = db.session.get(Evaluation, evaluation_id)
    if evaluation is None:
        abort(404)
    if not app.config["ENABLE_HISTORY"] and evaluation_id not in session.get("my_reports", []):
        abort(404)
    return evaluation


def _cached_report(app: Flask, key: str) -> Evaluation | None:
    """A recent AI-written report for the same resume and job, if one exists."""
    if app.config["CACHE_DAYS"] <= 0:
        return None
    since = utcnow() - timedelta(days=app.config["CACHE_DAYS"])
    matches = (
        Evaluation.query.filter(
            Evaluation.cache_key == key, Evaluation.status == STATUS_DONE, Evaluation.created_at >= since
        )
        .order_by(Evaluation.created_at.desc())
        .all()
    )
    # Don't reuse rule-based reports: an AI model may be available now.
    return next((e for e in matches if e.details_dict.get("source") == "ai"), None)


def _is_stale(app: Flask, evaluation: Evaluation) -> bool:
    """True if a background job should long since have finished (e.g. lost on restart)."""
    budget = (
        app.config["GROQ_TIMEOUT"] * max(1, len(app.config["GROQ_MODELS"]))
        + app.config["OLLAMA_TIMEOUT"]
        + 120
    )
    return evaluation.created_at is not None and utcnow() - evaluation.created_at > timedelta(seconds=budget)


def _previous_report(evaluation: Evaluation) -> Evaluation | None:
    """The same person's previous report for the same role, to show progress."""
    candidate = evaluation.candidate
    if candidate is None or evaluation.job is None or evaluation.created_at is None:
        return None
    query = (
        Evaluation.query.join(Candidate, Evaluation.candidate_id == Candidate.id)
        .join(Job, Evaluation.job_id == Job.id)
        .filter(
            Evaluation.id != evaluation.id,
            Evaluation.created_at <= evaluation.created_at,
            Evaluation.id < evaluation.id,
            Job.title == evaluation.job.title,
        )
    )
    if candidate.email:
        query = query.filter(Candidate.email == candidate.email)
    elif candidate.name and candidate.name != "Candidate":
        query = query.filter(Candidate.name == candidate.name)
    else:
        return None
    return query.order_by(Evaluation.id.desc()).first()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        flash(f"That file is too large. The limit is {app.config['MAX_UPLOAD_MB']} MB.", "error")
        return redirect(url_for("analyze_form"))

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("error.html", code=404, message="We couldn't find that page."), 404

    @app.errorhandler(500)
    def server_error(_exc):
        db.session.rollback()
        return render_template("error.html", code=500, message="Something went wrong on our side."), 500


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
