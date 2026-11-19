import json
from datetime import timedelta

from models import STATUS_DONE, STATUS_PENDING, Candidate, Evaluation, Job, db, utcnow
from services.advisor import Feedback, ground_feedback
from services.maintenance import purge_older_than
from services.scoring import score_resume, skills_from_text
from services.sections import split_sections
from services.worker import RateLimiter
from tests.conftest import SAMPLE_RESUME, csrf_token, upload
from tests.test_llm import GOOD_JSON, FakeProvider

JOB_POSTING = (
    "We are hiring a Backend Engineer to build Python microservices with FastAPI and PostgreSQL. "
    "You will deploy on Kubernetes and AWS, use Kafka for events and write tests with pytest. "
    "Experience with GraphQL is a plus."
)


def post_analysis(client, **data):
    form = {"csrf_token": csrf_token(client), "resume": upload(), **data}
    return client.post("/analyze", data=form, content_type="multipart/form-data")


def job_id(app, title="Backend Developer"):
    with app.app_context():
        return Job.query.filter_by(title=title).first().id


def use_fake_llm(app, provider):
    from services.llm import LLMChain

    app.extensions["analyzer"]["llm"] = LLMChain([provider])


# --- sections & evidence ------------------------------------------------------

def test_split_sections_finds_headings():
    sections = split_sections(SAMPLE_RESUME)
    assert {"summary", "experience", "projects", "skills", "education"} <= set(sections)
    assert "Redis-backed" in sections["projects"]
    assert split_sections("no headings at all here\njust text") == {}


def test_listed_only_skills_get_partial_credit():
    resume = "Experience\nBuilt Flask APIs in Python.\n\nSkills\nPython, Flask, Rust"
    b = score_resume(resume, "Python Flask Rust", "Python;Flask;Rust")
    assert b.matched_skills == ["Python", "Flask", "Rust"]
    assert b.listed_only_skills == ["Rust"]
    assert b.skill_score == round((2 + 0.6) / 3 * 100)


def test_skills_from_text_uses_catalogue_vocabulary():
    vocab = ["Python", "FastAPI", "PostgreSQL", "Kafka|RabbitMQ", "Kubernetes|k8s", "Scripting=Python|Bash", "Go"]
    skills = skills_from_text(JOB_POSTING, vocab)
    names = [s.split("|")[0] for s in skills.split(";")]
    assert names == ["Python", "FastAPI", "PostgreSQL", "Kubernetes", "Kafka"]  # order of first mention
    assert "Scripting" not in skills  # umbrella entries are skipped


def test_skills_from_text_names_the_tool_actually_mentioned():
    vocab = [
        "Jenkins|GitHub Actions|GitLab CI",
        "Kubernetes|k8s",
        "Experimentation|experiments",
        "Web framework=Flask|FastAPI",
    ]
    skills = skills_from_text("CI with GitHub Actions on k8s; run experiments; APIs in FastAPI.", vocab).split(";")
    assert skills == ["GitHub Actions", "k8s", "FastAPI"]


# --- grounding ------------------------------------------------------------------

def test_grounding_removes_contradictions_and_backfills():
    breakdown = score_resume(SAMPLE_RESUME, "Python Kafka pytest", "Python;Kafka;pytest")
    fb = Feedback(
        summary="ok",
        strengths=["Strong Kafka streaming experience.", "Solid Python APIs."],
        gaps=["pytest is not mentioned in the resume.", "No evidence of Kafka."],
        recommendations=[
            "List Kafka in your skills section.",
            "If you have used Kafka, add a project bullet about it.",
            "Quantify impact in each bullet.",
        ],
    )
    out = ground_feedback(fb, SAMPLE_RESUME, breakdown, "Python;Kafka;pytest", job_title="Backend Engineer")
    assert out.strengths == ["Solid Python APIs."]
    assert out.gaps == ["No evidence of Kafka."]
    assert "List Kafka in your skills section." not in out.recommendations
    assert "If you have used Kafka, add a project bullet about it." in out.recommendations
    assert len(out.recommendations) == 5 and out.grounding_removed == 3


# --- pasted job descriptions ----------------------------------------------------

def test_custom_job_posting_flow(app, client):
    resp = post_analysis(client, mode="custom", custom_title="Backend Engineer at Acme", custom_description=JOB_POSTING)
    assert resp.status_code == 302
    html = client.get(resp.headers["Location"]).get_data(as_text=True)
    assert "Backend Engineer at Acme" in html and "pasted job posting" in html
    with app.app_context():
        job = Job.query.filter_by(is_custom=True).one()
        assert "FastAPI" in job.skills
    # Pasted jobs never show up in the catalogue.
    assert "Backend Engineer at Acme" not in client.get("/roles").get_data(as_text=True)
    assert "Backend Engineer at Acme" not in client.get("/analyze").get_data(as_text=True)


def test_custom_job_posting_too_short(client):
    resp = post_analysis(client, mode="custom", custom_description="Python dev")
    assert resp.status_code == 302 and "mode=custom" in resp.headers["Location"]


# --- background feedback ----------------------------------------------------------

def test_background_feedback_completes(tmp_path):
    from app import create_app

    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'bg.db'}",
            "ENABLE_EMBEDDINGS": False,
            "RAG_WARMUP": False,
            "GROQ_API_KEY": None,
            "OLLAMA_ENABLED": False,
            "ASYNC_FEEDBACK": True,
        }
    )
    use_fake_llm(app, FakeProvider("ollama/mistral:7b", result=GOOD_JSON))
    client = app.test_client()
    resp = post_analysis(client, job_id=job_id(app))
    eval_id = int(resp.headers["Location"].rsplit("/", 1)[1])

    worker = app.extensions["analyzer"]["worker"]
    worker._pool.shutdown(wait=True)  # wait for the background job

    status = client.get(f"/report/{eval_id}/status").get_json()
    assert status == {"status": STATUS_DONE, "ready": True}
    html = client.get(f"/report/{eval_id}").get_data(as_text=True)
    assert "Strong backend profile." in html and "mistral:7b (local Ollama)" in html
    with app.app_context():
        db.engine.dispose()


def test_pending_report_shows_progress_and_stale_jobs_finish(app, client):
    with app.app_context():
        cand = Candidate(name="P", resume_text=SAMPLE_RESUME)
        job = Job.query.filter_by(title="Backend Developer").first()
        breakdown = score_resume(SAMPLE_RESUME, job.description, job.skills)
        ev = Evaluation(candidate=cand, job=job, score=50, status=STATUS_PENDING,
                        details=json.dumps({"breakdown": breakdown.to_dict()}))
        db.session.add_all([cand, ev])
        db.session.commit()
        ev_id = ev.id

    html = client.get(f"/report/{ev_id}").get_data(as_text=True)
    assert "Writing your feedback" in html and "data-poll" in html
    assert client.get(f"/report/{ev_id}/status").get_json()["ready"] is False

    # Simulate a job lost on restart: it is finished with rule-based feedback.
    with app.app_context():
        ev = db.session.get(Evaluation, ev_id)
        ev.created_at = utcnow() - timedelta(hours=2)
        db.session.commit()
    html = client.get(f"/report/{ev_id}").get_data(as_text=True)
    assert "Writing your feedback" not in html and "Recommended edits" in html


# --- caching, rate limit, delete, delta, retention ------------------------------

def test_identical_submission_reuses_ai_report(app, client):
    use_fake_llm(app, FakeProvider("openai/gpt-oss-120b", result=GOOD_JSON))
    first = post_analysis(client, job_id=job_id(app)).headers["Location"]
    second = post_analysis(client, job_id=job_id(app))
    assert second.headers["Location"] == first
    with app.app_context():
        assert Evaluation.query.count() == 1


def test_rule_based_reports_are_not_reused(app, client):
    first = post_analysis(client, job_id=job_id(app)).headers["Location"]
    second = post_analysis(client, job_id=job_id(app)).headers["Location"]
    assert first != second


def test_rate_limiter():
    limiter = RateLimiter(2, 600)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    assert RateLimiter(0, 600).allow("a")


def test_rate_limit_blocks_analysis(app, client):
    app.extensions["analyzer"]["limiter"] = RateLimiter(1, 600)
    post_analysis(client, job_id=job_id(app))
    resp = post_analysis(client, job_id=job_id(app))
    assert "mode=catalogue" in resp.headers["Location"]
    follow = client.get(resp.headers["Location"]).get_data(as_text=True)
    assert "lot of analyses" in follow


def test_delete_report_removes_candidate_and_custom_job(app, client):
    loc = post_analysis(client, mode="custom", custom_description=JOB_POSTING).headers["Location"]
    eval_id = int(loc.rsplit("/", 1)[1])
    resp = client.post(f"/report/{eval_id}/delete", data={"csrf_token": csrf_token(client)})
    assert resp.status_code == 302
    assert client.get(loc).status_code == 404
    with app.app_context():
        assert Evaluation.query.count() == 0 and Candidate.query.count() == 0
        assert Job.query.filter_by(is_custom=True).count() == 0


def test_delete_requires_csrf(app, client):
    loc = post_analysis(client, job_id=job_id(app)).headers["Location"]
    client.post(loc + "/delete", data={})
    assert client.get(loc).status_code == 200


def test_score_delta_against_previous_report(app, client):
    post_analysis(client, job_id=job_id(app))
    improved = SAMPLE_RESUME + "\nAlso built GraphQL APIs on MongoDB and Kafka."
    form = {"csrf_token": csrf_token(client), "job_id": job_id(app), "resume": upload(improved)}
    loc = client.post("/analyze", data=form, content_type="multipart/form-data").headers["Location"]
    html = client.get(loc).get_data(as_text=True)
    assert "since last report" in html and "▲ +" in html


def test_retention_purges_old_reports(app, client):
    post_analysis(client, job_id=job_id(app))
    with app.app_context():
        ev = Evaluation.query.one()
        ev.created_at = utcnow() - timedelta(days=40)
        db.session.commit()
        assert purge_older_than(30) == 1
        assert Evaluation.query.count() == 0 and Candidate.query.count() == 0
        assert purge_older_than(0) == 0
