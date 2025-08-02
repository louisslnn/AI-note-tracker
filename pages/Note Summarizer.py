import streamlit as st
from backend.summarizer import summarize_note

# --- SUMMARIZE NOTE ---
st.header("Summarize your note")
note_in = st.text_area("Write your note here...")

if st.button("Summarize note"):
    summarized_note = summarize_note(str(note_in))
    st.markdown(f"> {summarized_note}")