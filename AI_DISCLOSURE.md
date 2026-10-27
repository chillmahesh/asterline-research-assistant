# AI-Use Disclosure — Asterline Research Assistant

## 1. AI Tooling Utilized
In accordance with the assignment guidelines, this project utilized **Google Antigravity IDE** (powered by DeepMind advanced agentic coding models) as an AI pair-programming assistant and software engineering accelerator.

---

## 2. Scope of AI Assistance
The AI tooling assisted in the following areas:

1. **Workspace Inspection & Corpus Extraction**:
   - Inspected `Batch5_Candidate_Assignment_Complete.pdf` using `PyMuPDF` to discover the embedded attachments.
   - Cleanly extracted `corpus.jsonl` from the PDF attachment without altering any record, field, or byte.
   - Generated the SHA-256 hash calculation and schema validation scripts.

2. **Architecture & Multi-Agent Design**:
   - Structured the LangGraph state transitions across the three required agents: Research Agent (Agent A), Verification and Synthesis Agent (Agent B), and Memory Agent (Agent C).
   - Designed typed state dictionaries (`AgentState`) and bounded secondary retrieval conditions.

3. **Persistent Memory & Isolation Implementation**:
   - Drafted SQLite relational schemas for conversations, messages, and active/superseded memories.
   - Formulated storage-layer `user_id` scoping in every query to ensure that synthetic user contexts cannot leak.
   - Implemented regex-based privacy interceptors to prevent API keys and credentials from entering persistent storage.

4. **Automated Evaluation & Testing Suite**:
   - Created the benchmark dataset `evaluation/cases.json` covering all core scenarios specified in the assignment brief.
   - Authored deterministic evaluation metrics for retrieval recall, citation precision, groundedness, and conflict resolution.
   - Implemented comparative evaluation runners measuring baseline vs. multi-agent performance.

5. **Streamlit User Interface & Documentation**:
   - Built the web interface in `app.py` featuring user profile selection, chat history, expandable citation inspectors, and memory management tools.
   - Assisted in formatting `DESIGN.md`, `README.md`, and technical walkthrough notes.

---

## 3. Human Review & Verification Process
All AI-generated code, documentation, and evaluation assets were rigorously reviewed and empirically verified:

- **Corpus Integrity Verification**:
  - Validated that `corpus.jsonl` contains exactly 75 records across 25 unique documents.
  - Calculated and verified the SHA-256 hash (`f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1`), ensuring zero modifications were made to the original benchmark data.
- **Conflict & Supersession Ground Truth Check**:
  - Manually verified that `handbook:1` explicitly supersedes `runbook:0` for the on-call stipend ($175 replacing $140).
  - Manually verified that `retention:0` explicitly supersedes `security:1` for cold-storage retention (20 days replacing 60 days).
  - Confirmed that incident investigations (INC-2026-03 and INC-2025-12) reflect exact patch versions (2.3.1 and 2.1.2).
- **Execution & Test Validation**:
  - Ran automated test suites (`test_corpus.py`, `test_retrieval.py`, `test_citations.py`, `test_conflicts.py`, `test_memory.py`, `test_user_isolation.py`, `test_evaluation.py`) directly against real SQLite databases and the vector store.
  - Verified that no assertions were weakened or mocked to produce artificially passing results.
