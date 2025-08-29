from os.path import isfile, join
from os import listdir
from embedings import load_pdf, load_text, get_embedding_function, evaluate_embedding
from vectors import make_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


def run_vectorization(path="data"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    pages: List[Document] = []
    for file in onlyfiles:
        fp = join(path, file)
        if file.endswith(".pdf"):
            pages.extend(load_pdf(fp))

        elif file.endswith(".txt"):
            pages.extend(load_text(fp))

        else:
            print(f"Unsupported file format: {file}")
    
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