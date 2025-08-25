import os

def load_pdf(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def load_text(file_path):
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # Load plain text file
    loader = TextLoader("notes.txt", encoding="utf-8")
    pages = loader.load()
    return pages


def get_embedding_function():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])

def evaluate_embedding(prediction, reference):
    from langchain.evaluation import load_evaluator
    evaluator = load_evaluator("embedding_distance", embeddings=get_embedding_function())
    return evaluator.evaluate_strings(prediction=prediction, reference=reference)
