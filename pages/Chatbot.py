import streamlit as st
import time
from backend.database_handler import retrieve_answer_from_notes

# ---------------------
# Animation Functions
# ---------------------

def animated_dots(duration=3):
    """Show animated thinking dots while waiting for a response."""
    dot_area = st.empty()
    start_time = time.time()
    while time.time() - start_time < duration:
        for dots in ["", ".", "..", "...", "..", "."]:
            dot_area.markdown(f"**Thinking{dots}**")
            time.sleep(0.3)
    dot_area.empty()

def simulate_typing(text, speed=0.005):
    """Simulate typing effect character by character."""
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(speed)

# ---------------------
# Page Config & Header
# ---------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🧠")
st.title("🧠 AI Smart Knowledge Tracker - Chatbot")

st.markdown("This chatbot uses the **Llama3.2** model to answer questions based on your personal notes.")

# ---------------------
# Chat History
# ---------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------
# Chat Input + Response
# ---------------------

if prompt := st.chat_input("Ask a question about your notes..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        animated_dots(duration=3)
        response = retrieve_answer_from_notes(prompt)
        simulate_typing(response, speed=0.01)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
