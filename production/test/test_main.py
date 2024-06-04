from fastapi.testclient import TestClient
from vkr_prod.main import app

client = TestClient(app)


def test_generate_answer():
    response = client.post("/generate-answer", json={"title": "Test Title", "text": "Test Text"})
    assert response.status_code == 200
    assert "response" in response.json()
