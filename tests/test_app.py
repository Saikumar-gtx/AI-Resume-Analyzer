from models import Evaluation, Job, db
from tests.conftest import csrf_token, upload


def first_job_id(app, title="Backend Developer"):
    with app.app_context():
        return Job.query.filter_by(title=title).first().id


def test_home_page(client):
    html = client.get("/").get_data(as_text=True)
    assert "Target roles" in html and "35" in html


def test_analyze_form_lists_jobs_and_preselects(app, client):
    job_id = first_job_id(app)
    html = client.get(f"/analyze?job={job_id}").get_data(as_text=True)
    assert "Backend Developer" in html and "Aerospace Engineer" in html
    assert f'value="{job_id}" data-category="Software" selected' in html


def test_roles_page(client):
    html = client.get("/roles").get_data(as_text=True)
    assert "35 roles" in html and 'data-category="Data &amp; AI"' in html


def test_report_tab_disabled_until_first_report(client):
    assert "is-disabled" in client.get("/").get_data(as_text=True)


def test_health(client):
    data = client.get("/health").get_json()
    assert data["status"] == "ok" and data["jobs"] >= 30 and data["ai_feedback"] is False


def test_analyze_end_to_end(app, client):
    resp = client.post(
        "/analyze",
        data={
            "csrf_token": csrf_token(client),
            "job_id": first_job_id(app),
            "candidate_name": "  Jane   Doe ",
            "resume": upload(),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302 and "/report/" in resp.headers["Location"]

    assert "is-disabled" not in client.get("/").get_data(as_text=True)
    report = client.get(resp.headers["Location"])
    html = report.get_data(as_text=True)
    assert report.status_code == 200
    assert "Jane Doe" in html and "Recommended edits" in html and "automated checks" in html

    with app.app_context():
        evaluation = db.session.get(Evaluation, 1)
        assert evaluation.candidate.email == "jane.doe@example.com"
        assert evaluation.details_dict["breakdown"]["matched_skills"]
        assert 0 <= evaluation.score <= 100


def test_analyze_requires_csrf(app, client):
    client.get("/")
    resp = client.post(
        "/analyze",
        data={"job_id": first_job_id(app), "resume": upload()},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "session expired" in resp.get_data(as_text=True)


def test_analyze_rejects_bad_extension(app, client):
    resp = client.post(
        "/analyze",
        data={"csrf_token": csrf_token(client), "job_id": first_job_id(app), "resume": upload(name="resume.exe")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Unsupported file type" in resp.get_data(as_text=True)


def test_analyze_rejects_missing_job(client):
    resp = client.post(
        "/analyze",
        data={"csrf_token": csrf_token(client), "job_id": "9999", "resume": upload()},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "select a target job" in resp.get_data(as_text=True)


def test_flash_message_rendered_once(app, client):
    resp = client.post(
        "/analyze",
        data={"csrf_token": csrf_token(client), "job_id": first_job_id(app), "resume": upload("short")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.get_data(as_text=True).count("enough readable text") == 1


def test_unknown_report_404(client):
    assert client.get("/report/12345").status_code == 404


def test_history_page(client):
    assert client.get("/history").status_code == 200


def test_history_can_be_disabled(tmp_path):
    from app import create_app

    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'h.db'}",
            "ENABLE_EMBEDDINGS": False,
            "RAG_WARMUP": False,
            "ENABLE_HISTORY": False,
        }
    )
    assert app.test_client().get("/history").status_code == 404
    with app.app_context():
        db.engine.dispose()
