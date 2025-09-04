from os.path import isfile
from embedings import load_pdf, load_text, load_docx
from vectors import make_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import glob


def run_vectorization(path="data"):
    only_files = glob.glob(f"{path}/**/*", recursive=True)
    only_files = [f for f in only_files if isfile(f)]
    print(f"Found {len(only_files)} files in {path}, {only_files}")
    pages: List[Document] = []
    for file in only_files:
        fp = file
        if file.endswith(".pdf"):
            pages.extend(load_pdf(fp))

        elif file.endswith(".txt"):
            pages.extend(load_text(fp))

        elif file.endswith(".docx"):
            pages.extend(load_docx(fp))

    if not pages:
        raise ValueError(f"No supported files found in {path}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50, 
        length_function=len, 
        separators=["\n\n", "\n", " "]
    )

    chunks = splitter.split_documents(pages)
    global vector_store
    vector_store = make_vectors(chunks)
    print(f"Loaded {len(chunks)} chunks")
    return vector_store