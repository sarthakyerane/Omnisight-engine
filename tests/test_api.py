"""
Tests for the Query API (api/routes/query.py and api/routes/frames.py)

Uses FastAPI's TestClient and mocks the embeddings module and DB session
so no real ChromaDB or SQLite is needed.

Covers:
- GET /health returns 200
- GET /query with valid q returns QueryResponse structure
- GET /query with missing q returns 422
- GET /query with q too short returns 422
- GET /frames returns paginated list
- GET /frames/{id} returns 404 for unknown frame
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestQueryEndpoint:
    def test_missing_query_param_returns_422(self):
        response = client.get("/query")
        assert response.status_code == 422

    def test_too_short_query_returns_422(self):
        response = client.get("/query?q=a")
        assert response.status_code == 422

    def test_valid_query_returns_correct_structure(self):
        mock_results = [
            {
                "frame_id": 1,
                "distance": 0.1234,
                "summary": "User was editing Python code in VSCode.",
                "app_name": "code.exe",
                "ts": "2024-01-15T10:30:00+00:00",
                "tags": "coding, python, vscode",
            }
        ]
        with patch("api.routes.query.embeddings.query", return_value=mock_results):
            response = client.get("/query?q=python+code+editing")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "python code editing"
        assert data["total_results"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["frame_id"] == 1
        assert data["results"][0]["app_name"] == "code.exe"

    def test_embedding_error_returns_500(self):
        with patch("api.routes.query.embeddings.query", side_effect=RuntimeError("DB error")):
            response = client.get("/query?q=something+went+wrong")
        assert response.status_code == 500

    def test_empty_results_returns_valid_response(self):
        with patch("api.routes.query.embeddings.query", return_value=[]):
            response = client.get("/query?q=no+results+here")
        assert response.status_code == 200
        assert response.json()["total_results"] == 0
        assert response.json()["results"] == []


class TestFramesEndpoint:
    def test_get_frame_404_for_unknown_id(self):
        mock_db = MagicMock()
        mock_db.get.return_value = None

        from api.routes.frames import get_db
        app.dependency_overrides[get_db] = lambda: mock_db
        response = client.get("/frames/99999")
        app.dependency_overrides.clear()

        assert response.status_code == 404
        assert "99999" in response.json()["detail"]

    def test_list_frames_returns_empty_list_when_no_data(self):
        mock_db = MagicMock()
        # Simulate: count query returns 0, rows query returns []
        mock_db.execute.return_value.scalar_one.return_value = 0
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        from api.routes.frames import get_db
        app.dependency_overrides[get_db] = lambda: mock_db
        response = client.get("/frames")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["total"] == 0
        assert response.json()["frames"] == []
