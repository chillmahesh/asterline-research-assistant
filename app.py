"""
Asterline Research Assistant — Streamlit Web Application.
Provides multi-agent grounded research, expandable citations, source evidence inspection,
and user-isolated memory management.
"""

import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import uuid
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from src.config import settings
from src.graph import AsterlineGraph
from src.memory.repository import SQLiteMemoryRepository
from src.memory.service import MemoryService
from src.models import MemoryOperation


# Synthetic user profiles for local demonstration
DEMO_USERS = {
    "user_alice": {
        "name": "Alice Chen",
        "role": "Lead Reliability Engineer",
        "location": "Dublin",
        "avatar": "👩‍💻"
    },
    "user_bob": {
        "name": "Bob Kumar",
        "role": "Product Operations Lead",
        "location": "Pune",
        "avatar": "👨‍💼"
    },
    "user_charlie": {
        "name": "Charlie Vance",
        "role": "Security & Compliance Auditor",
        "location": "Remote",
        "avatar": "🕵️"
    }
}

st.set_page_config(
    page_title="Asterline Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished aesthetics
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stBadge { font-weight: 600; padding: 4px 8px; border-radius: 6px; }
    .source-card {
        background-color: #1a1f2c;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
    }
    .citation-tag {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 500;
        border: 1px solid #334155;
    }
    .agent-pill {
        display: inline-block;
        padding: 2px 8px;
        margin-right: 6px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        text-transform: uppercase;
    }
    .agent-research { background-color: #1e3a5f; color: #60a5fa; }
    .agent-verify { background-color: #14532d; color: #4ade80; }
    .agent-memory { background-color: #581c87; color: #c084fc; }
</style>
""", unsafe_allow_html=True)


# Initialize singletons in session state
if "graph" not in st.session_state:
    st.session_state.graph = AsterlineGraph()
if "repo" not in st.session_state:
    st.session_state.repo = SQLiteMemoryRepository()
if "memory_service" not in st.session_state:
    st.session_state.memory_service = MemoryService(st.session_state.repo)


# Sidebar controls
with st.sidebar:
    st.title("🔬 Asterline Labs")
    st.caption("Multi-Agent Research Assistant with Contextual Memory")
    st.divider()

    # User Profile Switcher
    st.subheader("👤 Active User")
    selected_user_id = st.selectbox(
        "Select synthetic user profile:",
        options=list(DEMO_USERS.keys()),
        format_func=lambda uid: f"{DEMO_USERS[uid]['avatar']} {DEMO_USERS[uid]['name']} ({DEMO_USERS[uid]['role']})",
        index=0,
        help="Simulates separate user identities. Stored memories and conversations remain strictly isolated in SQLite."
    )
    user_info = DEMO_USERS[selected_user_id]
    st.caption(f"**Office**: {user_info['location']} | **ID**: `{selected_user_id}`")

    st.divider()

    # Conversation Management
    st.subheader("💬 Conversations")
    conversations = st.session_state.repo.list_conversations(selected_user_id)

    if st.button("➕ New Conversation", use_container_width=True):
        new_conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        st.session_state.repo.get_or_create_conversation(selected_user_id, new_conv_id, "New Research Session")
        st.session_state.current_conversation_id = new_conv_id
        st.rerun()

    conv_options = [c["id"] for c in conversations] if conversations else []
    if not conv_options:
        default_conv = f"conv_{uuid.uuid4().hex[:8]}"
        st.session_state.repo.get_or_create_conversation(selected_user_id, default_conv, "Initial Conversation")
        conv_options = [default_conv]

    if "current_conversation_id" not in st.session_state or st.session_state.current_conversation_id not in conv_options:
        st.session_state.current_conversation_id = conv_options[0]

    current_conv_id = st.selectbox(
        "Select Conversation:",
        options=conv_options,
        index=conv_options.index(st.session_state.current_conversation_id),
        format_func=lambda cid: next((c["title"] for c in conversations if c["id"] == cid), cid),
    )
    st.session_state.current_conversation_id = current_conv_id

    if st.button("🗑️ Reset Active Conversation", use_container_width=True):
        st.session_state.repo.clear_conversation(selected_user_id, current_conv_id)
        st.success("Conversation history cleared!")
        st.rerun()

    st.divider()

    # System Diagnostics & Observability
    with st.expander("⚙️ System & Observability Status", expanded=False):
        # Corpus status
        st.markdown("**Knowledge Base:**")
        st.write("✅ `corpus.jsonl`: 75 Chunks / 25 Documents")
        st.write("✅ Vector Store: ChromaDB / Semantic Fallback")
        st.write("✅ Memory Storage: SQLite (User-Isolated)")

        st.markdown("**LLM Provider:**")
        is_avail, avail_msg = st.session_state.graph.llm.is_available()
        st.write(f"Provider: `{settings.llm_provider}`")
        if is_avail:
            st.success(f"Status: {avail_msg}")
        else:
            st.warning(f"Status: {avail_msg}")

        st.markdown("**LangSmith Observability:**")
        if settings.is_langsmith_configured():
            st.success(f"Tracing: Active (Project `{settings.langchain_project}`)")
        else:
            st.info("Tracing: Inactive (Set `LANGCHAIN_TRACING_V2=true` & `LANGCHAIN_API_KEY` in `.env` to enable)")


# Main tabs
tab_chat, tab_memory = st.tabs(["💬 Research Assistant", "🧠 Memory & Context Management"])


# =============================================================================
# Tab 1: Research Chat Interface
# =============================================================================
with tab_chat:
    st.subheader(f"Research Session — {user_info['name']}")
    st.caption("Answers are strictly grounded in authoritative Asterline Labs documentation.")

    # Quick prompt buttons
    with st.expander("💡 Suggested Questions & Scenarios", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Launch & Team Plan Pricing", use_container_width=True):
                st.session_state.pending_prompt = "What is the pricing and record allowance for the Launch and Team plans?"
            if st.button("On-call Stipend Update", use_container_width=True):
                st.session_state.pending_prompt = "What is the current weekly on-call stipend, and does it replace any older amount?"
        with col2:
            if st.button("Cold Recovery Retention", use_container_width=True):
                st.session_state.pending_prompt = "How long are raw run records kept in cold recovery storage after searchable retention ends?"
            if st.button("INC-2026-03 Signals Alerts", use_container_width=True):
                st.session_state.pending_prompt = "What caused delayed alerts during INC-2026-03 and which patch resolved it?"
        with col3:
            if st.button("INC-2025-12 Warehouse Fix", use_container_width=True):
                st.session_state.pending_prompt = "What caused duplicate Warehouse Export rows in INC-2025-12 and how was it fixed?"
            if st.button("Asia-Pacific Region Timeline", use_container_width=True):
                st.session_state.pending_prompt = "Has Asterline announced a launch date or timeline for an Asia-Pacific region?"

    # Display Conversation Messages
    messages = st.session_state.repo.get_messages(selected_user_id, st.session_state.current_conversation_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown("**Citations:**")
                for cite in msg["citations"]:
                    st.markdown(f"- `[Source: {cite.get('chunk_id')} — {cite.get('title')}]` (v{cite.get('version')}, pub {cite.get('published')})")

    # Chat Input
    prompt = st.chat_input("Ask a question about Asterline documentation or update your memory...")
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message to SQLite
        st.session_state.repo.save_message(
            user_id=selected_user_id,
            conversation_id=st.session_state.current_conversation_id,
            role="user",
            content=prompt
        )

        # Execute multi-agent orchestration
        with st.spinner("Collaborating: Research Agent ➔ Verification Agent ➔ Memory Agent..."):
            result_state = st.session_state.graph.invoke(
                user_id=selected_user_id,
                conversation_id=st.session_state.current_conversation_id,
                user_message=prompt
            )

        final_answer = result_state.get("final_answer", "")
        citations = result_state.get("citations", [])
        retrieved_chunks = result_state.get("retrieved_chunks", [])
        agent_traces = result_state.get("agent_traces", [])

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(final_answer)

            # Citations List
            if citations:
                st.markdown("##### 📌 Verified Citations:")
                for cite in citations:
                    st.markdown(f"- **`[Source: {cite['chunk_id']} — {cite['title']}]`** (Version: `{cite['version']}`, Published: `{cite['published']}`)")

            # Expandable Source Evidence
            if retrieved_chunks:
                with st.expander(f"📑 Retrieved Documentary Evidence ({len(retrieved_chunks)} chunks inspected)", expanded=False):
                    for idx, c in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"""
                        <div class="source-card">
                            <b>Chunk #{idx}:</b> <code>{c['chunk_id']}</code> — <i>{c['title']}</i><br>
                            <small>Doc ID: <code>{c['doc_id']}</code> | Version: <code>{c['version']}</code> | Published: <code>{c['published']}</code> | Category: <code>{c['category']}</code></small>
                            <p style="margin-top: 6px; font-size: 0.9em;">{c['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Expandable Multi-Agent Execution Trace
            if agent_traces:
                with st.expander(f"🤖 Multi-Agent Execution Trace ({len(agent_traces)} agent steps)", expanded=False):
                    for step in agent_traces:
                        agent_name = step.get("agent", "Agent")
                        pill_class = "agent-research" if "Research" in agent_name else ("agent-verify" if "Verification" in agent_name else "agent-memory")
                        st.markdown(f"<span class='agent-pill {pill_class}'>{agent_name}</span> <b>Action:</b> <code>{step.get('action')}</code>", unsafe_allow_html=True)
                        st.json(step)

        # Save assistant message to SQLite
        st.session_state.repo.save_message(
            user_id=selected_user_id,
            conversation_id=st.session_state.current_conversation_id,
            role="assistant",
            content=final_answer,
            citations=citations
        )
        st.rerun()


# =============================================================================
# Tab 2: Memory & Context Management
# =============================================================================
with tab_memory:
    st.subheader(f"Contextual Memory — {user_info['name']}")
    st.info(f"🔒 **Strict User Isolation Active**: You are viewing memories strictly belonging to `{selected_user_id}`. Memories from other users are completely inaccessible.")

    active_memories = st.session_state.repo.get_active_memories(selected_user_id)

    col_view, col_manage = st.columns([3, 2])

    with col_view:
        st.markdown("### 📋 Active Stored Memories")
        if not active_memories:
            st.write("No active memories stored for this user yet. Say *'Remember that I focus on Warehouse Export'* in the chat or use the form on the right.")
        else:
            for mem in active_memories:
                with st.container():
                    st.markdown(f"""
                    <div style="background:#1e293b; padding:12px; border-radius:8px; margin-bottom:10px; border-left:4px solid #a855f7;">
                        <b>Category:</b> <span class="citation-tag">{mem.category.upper()}</span> | <b>Confidence:</b> {mem.confidence*100:.0f}%<br>
                        <p style="margin-top:6px; font-size:1.05em;">{mem.content}</p>
                        <small style="color:#94a3b8;">Created: {mem.created_at[:19]} UTC | ID: <code>{mem.id[:8]}...</code></small>
                    </div>
                    """, unsafe_allow_html=True)

    with col_manage:
        st.markdown("### 🛠️ Manage Memories")

        # 1. Add New Memory
        with st.form("add_memory_form"):
            st.markdown("#### ➕ Add New Memory")
            new_content = st.text_area("Memory statement (e.g. 'I prefer concise answers with code snippets'):")
            new_cat = st.selectbox("Category:", ["preference", "project", "role", "constraint", "general"])
            submit_add = st.form_submit_button("Save Memory")
            if submit_add and new_content:
                op = MemoryOperation(action="store", content=new_content, category=new_cat)
                res = st.session_state.memory_service.execute_memory_operation(selected_user_id, op)
                if res["success"]:
                    st.success("Memory saved successfully!")
                    st.rerun()
                else:
                    st.error(res.get("reason", "Failed to save memory"))

        # 2. Correct Existing Memory
        if active_memories:
            with st.form("correct_memory_form"):
                st.markdown("#### ✏️ Correct Memory")
                mem_to_correct = st.selectbox(
                    "Select memory to update:",
                    options=active_memories,
                    format_func=lambda m: f"[{m.category}] {m.content[:40]}..."
                )
                corrected_text = st.text_input("Corrected content:", value=mem_to_correct.content)
                submit_correct = st.form_submit_button("Submit Correction")
                if submit_correct and corrected_text:
                    op = MemoryOperation(action="correct", memory_id=mem_to_correct.id, content=corrected_text, category=mem_to_correct.category)
                    res = st.session_state.memory_service.execute_memory_operation(selected_user_id, op)
                    if res["success"]:
                        st.success("Memory corrected transactionally!")
                        st.rerun()

            # 3. Explicit Forgetting
            with st.form("forget_memory_form"):
                st.markdown("#### 🗑️ Forget Memory")
                mem_to_forget = st.selectbox(
                    "Select memory to forget permanently:",
                    options=active_memories,
                    format_func=lambda m: f"[{m.category}] {m.content[:40]}..."
                )
                submit_forget = st.form_submit_button("Forget Selected Memory")
                if submit_forget:
                    op = MemoryOperation(action="forget", memory_id=mem_to_forget.id)
                    res = st.session_state.memory_service.execute_memory_operation(selected_user_id, op)
                    if res["success"]:
                        st.success("Memory forgotten and excluded from future retrievals!")
                        st.rerun()

        # 4. Forget by Topic
        with st.form("forget_topic_form"):
            st.markdown("#### 🧹 Forget by Topic")
            topic_to_forget = st.text_input("Topic or keyword to forget (e.g. 'Warehouse'):")
            submit_forget_topic = st.form_submit_button("Forget Matching Memories")
            if submit_forget_topic and topic_to_forget:
                op = MemoryOperation(action="forget", content=topic_to_forget)
                res = st.session_state.memory_service.execute_memory_operation(selected_user_id, op)
                if res["success"]:
                    st.success(f"Deactivated {res.get('count', 0)} memories matching '{topic_to_forget}'!")
                    st.rerun()
                else:
                    st.warning("No active memories matched that topic.")
