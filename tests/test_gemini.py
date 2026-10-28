"""
Tests for Google Gemini LLM provider and LangSmith configuration.
"""

from src.config import settings
from src.agents.llm_client import LLMClient


def test_gemini_llm_client_initialization():
    client = LLMClient(provider="gemini")
    assert client.provider == "gemini"
    assert client.gemini_key is not None
    is_avail, msg = client.is_available()
    assert is_avail is True
    assert "Gemini API configured" in msg


def test_langsmith_settings_configured():
    assert settings.is_langsmith_configured() is True
    assert settings.langchain_project == "asterline-research-assistant"
    assert settings.langchain_endpoint == "https://api.smith.langchain.com"
