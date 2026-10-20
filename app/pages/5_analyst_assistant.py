"""Page 5: AI Retail Analyst Assistant - Tool-Grounded Natural Language Querying."""

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

# Optional API Key input in sidebar
st.sidebar.header("🔑 Gemini API Settings (Optional)")
user_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Optional: If left blank, the assistant runs in deterministic local mode.",
)
if user_api_key:
    st.sidebar.success("Custom Gemini API key active (Function Calling Mode).")
else:
    st.sidebar.info("Running in deterministic local tool-dispatch mode.")

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
