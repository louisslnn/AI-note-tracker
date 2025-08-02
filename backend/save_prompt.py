import os
import uuid

path_to_data_folder = "/Users/louissalanon/Desktop/AI:DS projects/smart-knowledge-tracker/notes_data/"

def save_prompt_as_text(prompt: str) -> None:
    note_id = str(uuid.uuid4())
    filename = f"{note_id}.txt"
    filepath = os.path.join(path_to_data_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(prompt)

#save_prompt_as_text("I got some really nice dick over here")
