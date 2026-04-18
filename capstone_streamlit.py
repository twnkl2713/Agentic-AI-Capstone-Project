import streamlit as st
import uuid
from agent import app, collection

st.set_page_config(page_title="HR Policy Bot", page_icon="🏢", layout="centered")
st.title("🏢 ABC Corporation HR Policy Bot")
st.caption("Your 24/7 Intelligent HR Assistant | Ask me about company policies")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# Knowledge base topics
KB_TOPICS = [
    "Casual Leave", "Sick Leave", "Earned Leave", "Maternity Leave",
    "Paternity Leave", "Attendance Policy", "Notice Period", "Work From Home",
    "Performance Review", "Travel Reimbursement", "Code of Conduct", "Leave Without Pay"
]

# Sidebar
with st.sidebar:
    st.header("🏢 ABC Corporation")
    st.write("Intelligent HR Assistant")
    st.divider()
    st.write("**📋 HR Policies Covered:**")
    for t in KB_TOPICS:
        st.write(f"• {t}")
    st.divider()
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.write(f"**Knowledge Base:** {collection.count()} documents")
    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.divider()
    st.caption("**Contact HR:** hr@company.com | Ext: 1234")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about HR policies..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke(
                {
                    "question": prompt,
                    "messages": st.session_state.messages[:-1],
                    "route": "",
                    "retrieved": "",
                    "sources": [],
                    "tool_result": "",
                    "answer": "",
                    "faithfulness": 0.0,
                    "eval_retries": 0,
                    "employee_name": ""
                },
                config=config
            )
            answer = result.get("answer", "Sorry, I couldn't process that request.")
        st.write(answer)

        sources = result.get("sources", [])
        if sources:
            with st.expander("📚 Sources used"):
                for s in sources:
                    st.write(f"• {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})