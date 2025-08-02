import os
from typing import List

path_to_data_folder = "/Users/louissalanon/Desktop/AI:DS projects/smart-knowledge-tracker/notes_data/"

def get_last_note() -> List:
    """
    This function goes through all files of the notes_data folder and returns an array with 
    """
    files = [os.path.join(path_to_data_folder, f) for f in os.listdir(path_to_data_folder) if f.endswith(".txt")]
    if not files:
        return None
    
    files.sort(key=os.path.getmtime)
    with open(files[-1], "r", encoding="utf-8") as f:
        return f.read()

def get_note(path_to_note: str):
    notepath = os.path.join(path_to_data_folder, path_to_note)
    with open(notepath, "r", encoding="utf-8") as f:
        return f.read()

#print(get_last_note(path_to_data_folder))