import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "rag_system_ready" in data
    assert "vector_store_status" in data

def test_root_redirect():
    """Test that root redirects to docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307  # Temporary redirect

def test_config_endpoint():
    """Test the configuration endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, dict)
