from transformers import pipeline

def summarize_note(note: str):
    summarizer = pipeline("summarization")

    cleaned_summarized_note = summarizer(note)[0]['summary_text'].strip().replace(' .', '.')

    return cleaned_summarized_note

note = """
AI multi app features:
Configure a multipage app (add folder pages), in order to implement several features: implement a local AI chatbot (see tutorial for corresponding interface) that answers a specific question based on your notes (RAG),  Semantic Summarization (use Hugging Face pipeline function), let users upload images, audio, pdfs and extract text using OCR (e.g. Tesseract or pytesseract) to add new notes.
"""
