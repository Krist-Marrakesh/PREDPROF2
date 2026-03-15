from tests.conftest import login_cookie


def test_root_redirects_to_login(client):
    r = client.get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["location"]


def test_login_page_returns_200(client):
    r = client.get("/login")
    assert r.status_code == 200


def test_login_invalid_credentials(client):
    r = client.post("/login", data={"username": "testadmin", "password": "wrong"})
    assert r.status_code == 401


def test_login_admin_redirects_to_admin(client):
    r = client.post("/login", data={"username": "testadmin", "password": "pass123"})
    assert r.status_code == 302
    assert "/admin" in r.headers["location"]


def test_login_user_redirects_to_dashboard(client):
    r = client.post("/login", data={"username": "testuser", "password": "user123"})
    assert r.status_code == 302
    assert "/dashboard" in r.headers["location"]


def test_logout_clears_cookie(client):
    cookie = login_cookie(client, "testuser", "user123")
    r = client.get("/logout", headers={"cookie": cookie})
    assert r.status_code == 302


def test_admin_page_accessible_by_admin(client):
    cookie = login_cookie(client, "testadmin", "pass123")
    r = client.get("/admin", headers={"cookie": cookie})
    assert r.status_code == 200


def test_admin_page_forbidden_for_user(client):
    cookie = login_cookie(client, "testuser", "user123")
    r = client.get("/admin", headers={"cookie": cookie})
    assert r.status_code == 403


def test_dashboard_requires_auth(client):
    r = client.get("/dashboard")
    assert r.status_code == 401


def test_dashboard_accessible_authenticated(client):
    cookie = login_cookie(client, "testuser", "user123")
    r = client.get("/dashboard", headers={"cookie": cookie})
    assert r.status_code == 200
