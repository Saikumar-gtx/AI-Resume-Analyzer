import io
from datetime import timedelta

import numpy as np

from models import Candidate, Evaluation, Job, db, utcnow
from services.analysis import rank_roles
from services.concepts import concepts_for
from services.embeddings import Embedder
from services.maintenance import purge_orphans
from services.scoring import RELATED_CREDIT, score_resume
from tests.conftest import SAMPLE_RESUME, csrf_token, upload

DESCRIBED = """Experience
- Orchestrated services with Helm charts on EKS.
- Set up experiment tracking and model versioning for training runs.
- Fine-tuned BERT for sentiment analysis.

Skills
Python, Docker
"""


# --- related (described, not named) skills -----------------------------------

def test_related_skills_get_partial_credit_and_evidence():
    b = score_resume(DESCRIBED, "ML platform", "Kubernetes|k8s;MLOps;Transformers;TensorFlow")
    assert b.related_evidence == {"Kubernetes": "helm", "MLOps": "experiment tracking", "Transformers": "bert"}
    assert b.missing_skills == ["TensorFlow"]  # no false credit for a different framework
    assert b.skill_score == round(3 * RELATED_CREDIT / 4 * 100)


def test_named_skill_is_not_reported_as_related():
    b = score_resume("Experience\nRan Kubernetes clusters with Helm.", "k8s", "Kubernetes")
    assert b.matched_skills == ["Kubernetes"] and b.related_evidence == {}


def test_concepts_lookup_by_alias_and_excludes_own_aliases():
    phrases = concepts_for("Unit testing", ("unit test", "unit testing", "pytest"))
    assert "automated tests" in phrases and "pytest" not in phrases
    assert concepts_for("Underwater basket weaving", ("weaving",)) == ()


def test_related_skills_appear_in_report_and_rules(client, app):
    with app.app_context():
        job = Job.query.filter_by(title="AI Engineer").first()
        job_id = job.id
    form = {"csrf_token": csrf_token(client), "job_id": job_id,
            "resume": (io.BytesIO((DESCRIBED + "\nBuilt LLM apps.").encode()), "r.txt")}
    loc = client.post("/analyze", data=form, content_type="multipart/form-data").headers["Location"]
    html = client.get(loc).get_data(as_text=True)
    assert "Described, not named" in html and "Transformers ≈ “bert”" in html
    assert "Name the skills you already describe" in html


# --- ranking ------------------------------------------------------------------

def test_rank_roles_orders_best_first(app):
    with app.app_context():
        jobs = Job.query.filter_by(is_custom=False).all()
        ranked = rank_roles(SAMPLE_RESUME, jobs, None)
        scores = [b.overall for _, b in ranked]
        assert scores == sorted(scores, reverse=True) and len(ranked) == len(jobs)
        backend_like = {"Backend Developer", "Backend Engineer", "Software Engineer", "Full Stack Developer"}
        assert ranked[0][0].title in backend_like
        assert ranked[-1][0].title != "Backend Developer"


class FakeModel:
    """Deterministic stand-in for SentenceTransformer."""

    def encode(self, texts, **_):
        vecs = np.array([[t.lower().count("python") + 0.1, t.lower().count("chef") + 0.1] for t in texts])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_batch_similarities_match_single_calls():
    emb = Embedder("fake")
    emb._model = FakeModel()
    jobs = ["Python developer", "Pastry chef"]
    batch = emb.resume_job_similarities("Python APIs in Python.\n\nMore Python.", jobs)
    singles = [emb.resume_job_similarity("Python APIs in Python.\n\nMore Python.", j) for j in jobs]
    assert np.allclose(batch, singles) and batch[0] > batch[1]


def test_match_flow_ranks_and_opens_full_report(app, client):
    resp = client.post(
        "/match",
        data={"csrf_token": csrf_token(client), "resume": upload(), "candidate_name": "Jane"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302 and "/match/" in resp.headers["Location"]
    html = client.get(resp.headers["Location"]).get_data(as_text=True)
    assert "Where your resume" in html and "Full report" in html and html.count('class="rank-row') == 35

    cand_id = int(resp.headers["Location"].rsplit("/", 1)[1])
    with app.app_context():
        job_id = Job.query.filter_by(title="Backend Developer").first().id
    report = client.post(f"/match/{cand_id}/report", data={"csrf_token": csrf_token(client), "job_id": job_id})
    assert "/report/" in report.headers["Location"]
    with app.app_context():
        ev = Evaluation.query.one()
        assert ev.candidate_id == cand_id  # reuses the uploaded resume, no duplicate
        assert Candidate.query.count() == 1


def test_match_results_are_private_to_the_session(app, client):
    loc = client.post(
        "/match", data={"csrf_token": csrf_token(client), "resume": upload()}, content_type="multipart/form-data"
    ).headers["Location"]
    other = app.test_client()
    assert other.get(loc).status_code == 404


def test_match_rejects_bad_upload(client):
    resp = client.post(
        "/match",
        data={"csrf_token": csrf_token(client), "resume": upload(name="x.exe")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Unsupported file type" in resp.get_data(as_text=True)


def test_unused_uploads_are_purged_after_a_day(app, client):
    client.post("/match", data={"csrf_token": csrf_token(client), "resume": upload()},
                content_type="multipart/form-data")
    with app.app_context():
        assert purge_orphans() == 0  # fresh upload is kept
        cand = Candidate.query.one()
        cand.created_at = utcnow() - timedelta(hours=30)
        db.session.commit()
        assert purge_orphans() == 1 and Candidate.query.count() == 0
