# To run Streamlit app: python3 -m streamlit run smart-knowledge-tracker/app.py

import streamlit as st
from backend.handle_database import *
from backend.get_last_note import get_last_note, get_note
from backend.save_prompt import save_prompt_as_text
from backend.summarizer import summarize_note

st.set_page_config(page_title="Smart Knowledge Tracker", layout="wide")
st.title("🧠 AI Smart Knowledge Tracker")

# --- ADD NOTE ---
st.header("📝 Add a Note")
with st.form("add_note_form"):
    note_input = st.text_area("Write your note here...", placeholder="e.g. Learned about backpropagation in neural networks.")
    submitted_note = st.form_submit_button("Save Note")
    
    if submitted_note:
        if note_input.strip():
            save_prompt_as_text(note_input)
            add_note_to_database(path_to_note=None, last_note=True)
            st.success("✅ Note successfully saved to the database.")
        else:
            st.error("⚠️ Note cannot be empty.")

# --- RETRIEVE NOTE ---
st.header("🔍 Retrieve a Note")

# ADD MULTISELECT RETRIEVAL MODE: EITHER BY QUERY, LAST NOTE, OR BY KEYWORD

st.subheader("Retrieve note by query")

with st.form("query_note_form"):
    query_input = st.text_input("Enter your query...", placeholder="e.g. What did I learn today?")
    submitted_query = st.form_submit_button("Search")
    
    if submitted_query:
        if query_input.strip():
            raw_output = query_to_database(query_input)["documents"]
            cleaned_output = clean_raw_output(raw_output)
            st.success("✅ Retrieved note:")
            st.markdown(f"> {cleaned_output}")
        else:
            st.error("⚠️ Query cannot be empty.")

st.subheader("Retrieve last note submitted")
if st.button("Get last note"):
    st.success("✅ Retrieved note:")
    st.markdown(f"> {get_last_note()}")

# --- RESET DATABASE ---
st.header("🧨 Reset Database")
with st.form("reset_form"):
    reset_button = st.form_submit_button("Reset Notes Database")

    if reset_button:
        reset_collection()
        st.warning("⚠️ All notes have been deleted.")
