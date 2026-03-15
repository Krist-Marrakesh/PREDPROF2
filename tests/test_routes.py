from tests.conftest import login_cookie


def test_create_user_by_admin(client):
    cookie = login_cookie(client, "testadmin", "pass123")
    r = client.post("/admin/create-user", headers={"cookie": cookie}, data={
        "username": "newuser", "password": "pw", "first_name": "N", "last_name": "U", "role": "user",
    })
    assert r.status_code == 200
    assert "newuser" in r.text


def test_create_duplicate_user_shows_error(client):
    cookie = login_cookie(client, "testadmin", "pass123")
    r = client.post("/admin/create-user", headers={"cookie": cookie}, data={
        "username": "testuser", "password": "pw", "first_name": "X", "last_name": "Y", "role": "user",
    })
    assert r.status_code == 200
    assert "уже существует" in r.text


def test_analytics_endpoint_returns_json(client):
    cookie = login_cookie(client, "testuser", "user123")
    r = client.get("/api/analytics/data", headers={"cookie": cookie})
    assert r.status_code == 200
    assert "ready" in r.json()


def test_analytics_forbidden_without_auth(client):
    r = client.get("/api/analytics/data")
    assert r.status_code == 401


def test_analytics_page_returns_html(client):
    cookie = login_cookie(client, "testuser", "user123")
    r = client.get("/analytics", headers={"cookie": cookie})
    assert r.status_code == 200
    assert "chart" in r.text.lower()
