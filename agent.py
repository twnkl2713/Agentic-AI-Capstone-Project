"""
agent.py - HR Policy Bot Agent
Contains all agent logic: State, Nodes, Graph, Vector DB
Author: Twinkle Pal | Roll Number: 2305744
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import date
import re

# ============================================================
# INITIALIZATION
# ============================================================

# LLM - Using smaller model to avoid rate limits
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB with 12 HR policy documents
client = chromadb.Client()
try:
    client.delete_collection("capstone_kb")
except:
    pass
collection = client.create_collection("capstone_kb")

# 12 HR Policy Documents
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Casual Leave",
        "text": """Casual Leave Policy: Each employee receives 12 casual leave days per financial year (April to March). Casual leave requires manager approval submitted at least 1 day in advance. Leave cannot be taken in half-day increments. Unused casual leave does not carry forward to the next year. During probation period (first 3 months), casual leave is not applicable."""
    },
    {
        "id": "doc_002",
        "topic": "Sick Leave",
        "text": """Sick Leave Policy: Each employee receives 10 sick leave days per financial year. For 1-2 days of sick leave, manager approval is sufficient. For 3 or more consecutive days, a medical certificate from a registered doctor is required. Sick leave must be reported on the day of absence before 10 AM. Unused sick leave does not carry forward."""
    },
    {
        "id": "doc_003",
        "topic": "Earned Leave",
        "text": """Earned Leave Policy: After completing 1 year of service, employees earn 15 earned leave days per year. Earned leave can be carried forward to the next year up to a maximum of 30 days. Encashment of earned leave is allowed at the time of resignation up to 15 days. Approval from both manager and HR is required for earned leave of 5+ days."""
    },
    {
        "id": "doc_004",
        "topic": "Maternity Leave",
        "text": """Maternity Leave Policy: Female employees are entitled to 26 weeks of paid maternity leave. This can be taken up to 8 weeks before the expected delivery date. The remaining leave must be taken after delivery. A medical certificate confirming pregnancy is required. Adoption leave of 12 weeks is available for adopting a child under 3 months."""
    },
    {
        "id": "doc_005",
        "topic": "Paternity Leave",
        "text": """Paternity Leave Policy: Male employees are entitled to 15 days of paid paternity leave. This must be taken within 30 days of the child's birth. A copy of the birth certificate is required. Paternity leave cannot be combined with any other leave type."""
    },
    {
        "id": "doc_006",
        "topic": "Attendance Policy",
        "text": """Attendance Policy: Working hours are 9 AM to 6 PM, Monday through Friday. A grace period of 15 minutes is allowed. Three late arrivals in a month count as one half-day leave. Employees must swipe their access card at entry and exit. Monthly attendance reports are shared with managers on the 1st of each month."""
    },
    {
        "id": "doc_007",
        "topic": "Notice Period",
        "text": """Notice Period Policy: The notice period for resignation is 60 days. The company may choose to waive the notice period and pay salary in lieu. During the notice period, employees are expected to complete knowledge transfer. Unused earned leave can be encashed or used to reduce the notice period. Gardening leave may be granted at company discretion."""
    },
    {
        "id": "doc_008",
        "topic": "Work From Home",
        "text": """Work From Home Policy: Employees may work from home up to 2 days per week with manager approval. WFH is not permitted during the probation period. Employees working from home must be available on Slack from 9 AM to 6 PM. Internet reimbursement of ₹1000 per month is provided. Core hours require all team members to be online from 11 AM to 3 PM."""
    },
    {
        "id": "doc_009",
        "topic": "Performance Review",
        "text": """Performance Review Policy: Performance reviews are conducted quarterly in January, April, July, and October. Each review includes self-assessment, manager assessment, and a calibration meeting. Ratings range from 1 (needs improvement) to 5 (exceptional). Employees with two consecutive ratings of 1 are placed on a performance improvement plan."""
    },
    {
        "id": "doc_010",
        "topic": "Travel Reimbursement",
        "text": """Travel Reimbursement Policy: Domestic travel: economy class flights, AC 2-tier train, or ₹8 per kilometer for personal vehicle. Hotel budget: ₹3000 per night. Food allowance: ₹500 per day. International travel: business class for flights over 8 hours, hotel budget: $150 per night. All travel requires pre-approval from the department head. Claims must be submitted within 15 days of travel completion."""
    },
    {
        "id": "doc_011",
        "topic": "Code of Conduct",
        "text": """Code of Conduct: Employees must maintain professional dress code: formals Monday through Thursday, business casual on Friday. Working hours are 9 AM to 6 PM. Personal phone use is limited to break times. Confidential company information must not be shared externally. Harassment of any kind is strictly prohibited and will result in immediate termination."""
    },
    {
        "id": "doc_012",
        "topic": "Leave Without Pay",
        "text": """Leave Without Pay Policy: When all leave balances are exhausted, employees may request Leave Without Pay. LWP requires HR approval. Salary is deducted for LWP days. LWP exceeding 10 days in a year affects the performance rating. LWP during probation extends the probation period by the number of LWP days taken."""
    },
]

# Add documents to ChromaDB
texts = [d["text"] for d in DOCUMENTS]
embeddings = embedder.encode(texts).tolist()
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[d["id"] for d in DOCUMENTS],
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
)

print(f"✅ Knowledge base loaded: {collection.count()} documents")

# ============================================================
# STATE DEFINITION
# ============================================================

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    employee_name: str

# ============================================================
# NODE FUNCTIONS
# ============================================================

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    
    # Extract employee name if they say "my name is X"
    employee_name = state.get("employee_name", "")
    question = state["question"].lower()
    if "my name is" in question:
        match = re.search(r"my name is (\w+)", question)
        if match:
            employee_name = match.group(1)
    
    return {"messages": msgs, "employee_name": employee_name}

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
    
    prompt = f"""You are a router for an HR Policy Chatbot.

Available options:
- retrieve: search the HR knowledge base for policy questions about leave, attendance, benefits, notice period, travel, code of conduct
- memory_only: answer from conversation history (e.g., 'what did you just say?', 'tell me again')
- tool: use the date calculator tool when asked about 'how many days until', 'days remaining', 'leave balance calculation'

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    
    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    q_emb = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    question = state["question"].lower()
    tool_result = ""
    
    if "how many days" in question or "days until" in question or "days left" in question:
        match = re.search(r"until (\w+) (\d+)", question)
        if match:
            month, day = match.group(1), int(match.group(2))
            month_map = {
                "january":1, "february":2, "march":3, "april":4, "may":5, "june":6,
                "july":7, "august":8, "september":9, "october":10, "november":11, "december":12
            }
            if month.lower() in month_map:
                target_date = date(date.today().year, month_map[month.lower()], day)
                if target_date < date.today():
                    target_date = date(date.today().year + 1, month_map[month.lower()], day)
                days_left = (target_date - date.today()).days
                tool_result = f"There are {days_left} days until {month} {day}."
            else:
                tool_result = f"Could not understand the month: {month}"
        else:
            tool_result = "Please specify a date like 'until March 31'"
    elif "leave balance" in question or "how many leaves" in question:
        tool_result = "Based on standard accrual: Casual Leave: 8 days remaining, Sick Leave: 6 days remaining, Earned Leave: 12 days remaining. Check HRMS for exact balance."
    else:
        tool_result = "I can calculate days until a date. Example: 'How many days until March 31?'"
    
    return {"tool_result": tool_result}

def answer_node(state: CapstoneState) -> dict:
    question = state["question"]
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)
    
    if context:
        system_content = f"""You are an HR Policy Assistant for the company.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: "I don't have that information in the HR policy handbook. Please contact HR at hr@company.com or call extension 1234."
Do NOT add information from your training data.
Do NOT give legal advice or opinions.

{context}"""
    else:
        system_content = """You are an HR Policy Assistant. Answer based on the conversation history. If you don't know, say so clearly."""
    
    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."
    
    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))
    
    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

def eval_node(state: CapstoneState) -> dict:
    answer = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)
    
    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}
    
    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""
    
    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}

# ============================================================
# GRAPH ASSEMBLY
# ============================================================

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"

graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")
graph.add_edge("memory", "router")

graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})

graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")

graph.add_edge("answer", "eval")
graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
graph.add_edge("save", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

print("✅ Agent compiled successfully!")