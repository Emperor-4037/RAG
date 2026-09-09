"""
Shared pytest fixtures.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.main import create_app


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c
