"""
Configuration management for Asterline Research Assistant.
Loads settings from environment variables or .env file with safe defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    # Workspace & Data Paths
    base_dir: Path = BASE_DIR
    corpus_path: Path = BASE_DIR / os.getenv("CORPUS_PATH", "corpus.jsonl")
    data_dir: Path = BASE_DIR / "data"
    chroma_persist_dir: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    sqlite_db_path: Path = BASE_DIR / os.getenv("SQLITE_DB_PATH", "data/asterline.db")

    # LLM Configuration
    # Supported: 'gemini', 'ollama', 'openai', 'mock'
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Local Embeddings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # LangSmith Observability
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
    langchain_api_key: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "asterline-research-assistant")
    langchain_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    def __post_init__(self):
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Set LangSmith environment variables if configured
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

    def is_langsmith_configured(self) -> bool:
        """Check if LangSmith tracing is active and validly configured."""
        return bool(self.langchain_tracing_v2 and self.langchain_api_key and not self.langchain_api_key.startswith("your_"))

    def check_ollama_reachable(self) -> bool:
        """Ping Ollama API to check if the server is actively running."""
        try:
            import requests
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False


# Global singleton instance
settings = Settings()
