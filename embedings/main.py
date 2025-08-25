import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import pandas as pd


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

loader = PyPDFLoader('Research Paper 2406.09647.pdf')
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", " "])
chunks = splitter.split_documents(pages)


def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])
embedding_function = get_embedding_function()


evaluator = load_evaluator("embedding_distance", embeddings=embedding_function)
evaluator.evaluate_strings(prediction="Sri Lanka", reference="Beach")

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory="vectorstore")

retriever = vectorstore.as_retriever(search_type='similarity')
relevant_chunks = retriever.invoke('What is the title of the article?')

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.
{context}
---
Answer the question based on the above context: {question}
"""


while True:
    question = input("What is your question ?\n")

    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    response = llm.invoke(prompt)

    def format_doc(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    rag_chain = (
        { "context": retriever | format_doc, "question": RunnablePassthrough() }
        | prompt_template
        | llm
    )
    rag_chain.invoke(question)

    class AnswerWithSources(BaseModel):
        answer: str = Field(description="Answer to question")
        sources: str = Field(description="Full direct text chunk from the context used to answer the question")
        reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    class ExtractedInfo(BaseModel):
        paper_title: AnswerWithSources
        paper_summary: AnswerWithSources
        publication_year: AnswerWithSources
        paper_authors: AnswerWithSources
    dag_chain = (
        { "context": retriever | format_doc, "question": RunnablePassthrough() }
        | prompt_template
        | llm.with_structured_output(ExtractedInfo, strict=True)
    )


    structured_response = dag_chain.invoke(question)
    df = pd.DataFrame([structured_response.model_dump()])
    answer_row, source_row, reasoning_row = [], [], []
    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    print(answer_row)
    print(source_row)
    print(reasoning_row)
    print('\n')
