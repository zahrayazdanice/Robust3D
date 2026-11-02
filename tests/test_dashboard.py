from robust3d.dashboard import create_app


def test_dashboard_index_serves():
    app = create_app()
    response = app.server.test_client().get("/")
    assert response.status_code == 200
    assert b"Robust3D" in response.data

