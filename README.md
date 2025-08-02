# AI Note Tracker

🧠 **AI Smart Knowledge Tracker** is a Streamlit-based application designed to help users efficiently manage and retrieve their notes using AI-powered features. The app supports multiple input methods, including text, voice, and file uploads, and provides intelligent search capabilities.

## Features

- **Add Notes**:
  - **Text Input**: Write and save notes directly in the app.
  - **Voice Input**: Record voice notes (under development).
  - **File Upload**: Upload text files containing notes.

- **Retrieve Notes**:
  - Search notes by keywords or queries.
  - Retrieve the last submitted note.

- **AI-Powered Summarization**:
  - Summarize notes for quick insights (future enhancement).

- **Vector Store Integration**:
  - Notes are stored in a vector database for efficient retrieval.

## Installation

1. Clone the repository:
    ```git clone https://github.com/louisslnn/AI-note-tracker.git```
2. Install dependencies:
    ```pip install -r requirements.txt```

3. Run the application:
    ```python3 -m streamlit run Home.py```

## File Structure
- ```backend/```: Contains backend logic for handling notes, database operations, and AI functionalities.

- ```database_handler.py```: Manages vector store operations.
    - ```get_last_note.py```: Retrieves the last saved note.
    - ```save_prompt.py```: Saves notes to the database.
    - ```voice_input.py```: Handles voice recognition (under development).
    - ```notes_data/```: Stores note-related data.

- ```templates/```: Contains HTML templates for the app.

- ```utils.py```: Utility functions for the project.

## Usage

1. Launch the app using the command:

```python3 -m streamlit run Home.py```

2. Use the **Add Note** section to input notes via text, voice, or file upload.

3. Use the **Retrieve Note** section to search for notes by keywords or retrieve the last saved note.

## Requirements
- Python 3.7 or higher
- Streamlit
- Additional dependencies listed in ```requirements.txt```

## Future Enhancements
- Advanced AI summarization for notes.
- Improved voice input functionality.
- Enhanced search capabilities with semantic understanding.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments
Built with Streamlit.
Inspired by the need for efficient knowledge tracking and retrieval.
