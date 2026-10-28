"""
Pytest configuration and global fixtures.
Ensures testing runs deterministically with mock reasoning provider,
preventing quota exhaustion, network flakiness, or test suite delays.
"""

import os
from src.config import settings

# Force mock provider globally for test runner
os.environ["LLM_PROVIDER"] = "mock"
settings.llm_provider = "mock"
