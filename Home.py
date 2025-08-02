# To run Streamlit app: python3 -m streamlit run smart-knowledge-tracker/app.py

import streamlit as st
from backend.get_last_note import get_last_note, get_last_note_content, get_note
from backend.save_prompt import save_prompt_as_text
from backend.summarizer import summarize_note
from backend.database_handler import add_note_to_vector_store, reset_vector_store
from backend.voice_input import recognize_speech

st.set_page_config(page_title="Smart Knowledge Tracker", layout="wide")
st.title("🧠 AI Smart Knowledge Tracker")

# --- ADD NOTE ---
st.header("📝 Add a Note")
add_type = st.selectbox(
    "How would you like to add your note?",
    ("Text Input", "Voice Input", "File Upload")
)

if add_type == "Text Input":
    with st.form("add_note_form"):
        note_input = st.text_area("Write your note here...", placeholder="e.g. Learned about backpropagation in neural networks.")
        submitted_note = st.form_submit_button("Save Note")
        
        if submitted_note:
            if note_input.strip():
                save_prompt_as_text(note_input)
                add_note_to_vector_store(note_text=get_last_note_content(), source_name="Text Input Note")
                st.success("✅ Note successfully saved to the database.")
            else:
                st.error("⚠️ Note cannot be empty.")

elif add_type == "Voice Input":
    with st.expander("Voice Input Note Tracker (Under Development)"):
        st.write("Press the button below to start recording your voice note.")
        if st.button("Start Recording"):
            st.info("Recording...")
            note_content = recognize_speech()
            save_prompt_as_text(str(note_content))
            add_note_to_vector_store(note_text=get_last_note_content(), source_name="Voice Input Note")
            st.success("✅ Note successfully saved from voice input.")

elif add_type == "File Upload":
    uploaded_file = st.file_uploader("Upload a text file with your note", type=["txt"])
    if uploaded_file is not None:
        note_content = uploaded_file.read().decode("utf-8")
        if note_content.strip():
            save_prompt_as_text(note_content)
            add_note_to_vector_store(note_text=get_last_note_content(), source_name="File Input Note")
            st.success("✅ Note successfully saved from file.")
        else:
            st.error("⚠️ Uploaded file is empty.")

"""        
# --- RETRIEVE NOTE ---
st.header("🔍 Retrieve a Note")

# ADD MULTISELECT RETRIEVAL MODE: EITHER BY QUERY, LAST NOTE, OR BY KEYWORD

st.subheader("Retrieve note by kerword(s)")

with st.form("query_note_form"):
    query_input = st.text_input("Enter your query...", placeholder="e.g. left features to implement for AI tracker project")
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
"""

# --- RESET DATABASE ---
st.header("🧨 Reset Database")
with st.form("reset_form"):
    reset_button = st.form_submit_button("Reset Notes Database")

    if reset_button:
        reset_vector_store()
        st.warning("⚠️ All notes have been deleted.")
