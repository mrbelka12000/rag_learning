from os.path import isfile, join
from os import listdir
from embedings import load_pdf, load_text, get_embedding_function, evaluate_embedding
from vectors import make_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def run(path="data"):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    pages = []
    for file in onlyfiles:
        if file.endswith(".pdf"):
            pages = load_pdf(join(path, file))
        
        elif file.endswith(".txt"):
            pages = load_text(join(path, file))
        else:
            print(f"Unsupported file format: {file}")
    
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