"""Page 5: AI Retail Analyst Assistant - Tool-Grounded Natural Language Querying."""

import sys
from pathlib import Path

# Add repository root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.assistant.agent import AnalystAssistant

st.set_page_config(page_title="AI Analyst Assistant | StockSense", page_icon="🤖", layout="wide")

st.title("🤖 AI Analyst Assistant: Grounded Supply Chain Querying")
st.markdown("""
Ask operational questions about forecasting accuracy, store anomalies, or replenishment trade-offs.

> **Strict Operational Guardrail**: The Assistant is **strictly prohibited from generating or calculating numbers directly**.
> Every single metric, percentage, unit, and dollar figure is retrieved by executing deterministic query tools
> over precomputed artifacts.
""")

st.sidebar.header("⚡ Built-in AI Engine")

user_api_key = st.sidebar.text_input(
    "Custom Groq API Key (Optional)",
    type="password",
    help="Paste your Groq API Key (gsk_...) here, or save it in Streamlit Cloud Secrets.",
)

# Assistant probing for active engine (evaluates user input, secrets, and env)
probe_assistant = AnalystAssistant(api_key=user_api_key if user_api_key else None)
active_mode = getattr(probe_assistant, "active_provider", "deterministic")
if callable(active_mode):
    active_mode = active_mode()

if active_mode == "groq":
    if user_api_key:
        st.sidebar.success("🟢 **Custom Groq Key Active** (`llama-3.3-70b-versatile`)")
    else:
        st.sidebar.success("🟢 **Built-in Groq Active** (`llama-3.3-70b-versatile`)")
else:
    st.sidebar.info(
        "⚪ **Deterministic Local Mode**\n\n"
        "*(If you already saved Secrets in Cloud Settings, click **Rerun** or paste your `gsk_...` key above)*"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Suggested Prompts")
suggestions = [
    "Why did forecast error spike for store CA_1?",
    "Where does LightGBM lose to the baseline heuristic?",
    "What was the bottom-line financial impact of our replenishment policy?",
    "How does the model perform on intermittent vs smooth demand?",
]

for s in suggestions:
    if st.sidebar.button(s, key=f"btn_{s}"):
        st.session_state["analyst_input"] = s

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am StockSense's Senior Supply Chain Analyst. I am connected to our verified "
                "backtest metrics and replenishment simulation engines. Ask me about store errors, SKU failures, "
                "or inventory trade-offs!"
            ),
            "tools_called": [],
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_called"):
            with st.expander("🛠️ Deterministic Tools Executed (Audit Trail)"):
                for tc in msg["tools_called"]:
                    st.json(tc)

# Chat input
prompt = st.chat_input("Ask a question about store performance, SKU failures, or replenishment...")
if "analyst_input" in st.session_state and st.session_state["analyst_input"]:
    prompt = st.session_state["analyst_input"]
    st.session_state["analyst_input"] = None

if prompt:
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via AnalystAssistant
    assistant = AnalystAssistant(api_key=user_api_key if user_api_key else None)
    with st.chat_message("assistant"):
        with st.spinner("Executing deterministic analytical tools over artifacts..."):
            response = assistant.ask(prompt)
            answer_text = response.get("answer", "No answer found.")
            tools_called = response.get("tools_called", [])

            st.markdown(answer_text)

            if tools_called:
                with st.expander("🛠️ Deterministic Tools Executed (Audit Trail)"):
                    for tc in tools_called:
                        st.json(tc)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "tools_called": tools_called,
    })
