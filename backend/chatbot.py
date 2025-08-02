from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def get_answer_from_ollama(question):
    #Create a new Ollama object
    model = OllamaLLM(model="llama3.2")

    template = """
    You are an expert in answering questions about daily notes, especially when it comes to to-do lists and remarks on work, daily and personal life.

    Here are some relevant notes: {notes}

    Here is the question to answer: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | model

    result = chain.invoke({"notes": [], "question": question})

    return result
