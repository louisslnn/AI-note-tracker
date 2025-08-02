import os
import uuid

path_to_data_folder = "/Users/louissalanon/Desktop/AI:DS projects/smart-knowledge-tracker/AI-note-tracker/notes_data/"

def save_prompt_as_text(prompt: str) -> None:
    """
    Saves the given prompt string as a uniquely named text file in the notes data folder.

    Args:
        prompt (str): The text content to be saved.

    Returns:
        None
    """
    filename = f"{uuid.uuid4()}.txt"
    filepath = os.path.join(path_to_data_folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(prompt)
