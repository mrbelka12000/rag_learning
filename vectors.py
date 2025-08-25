def make_vectors(chunks):
    from langchain_community.vectorstores import Chroma
    from embedings import get_embedding_function
    vectorstore = Chroma.from_documents(documents=chunks, embedding=get_embedding_function(), persist_directory="vectorstore")
    return vectorstore