
def answer(question: str, retriever) -> dict:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model="gpt-4o-mini")
    relevant_chunks = retriever.invoke(question)

    PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.
{context}
---
Answer the question based on the above context: {question}
"""

    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    def format_doc(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
   # Build a Runnable chain:
    rag_chain = (
        {                     # map inputs
            "context": retriever | format_doc,   # retriever returns docs -> format to string
            "question": RunnablePassthrough()    # pass the user question through
        }
        | prompt_template                        # fill {context}, {question}
        | llm                                    # call the model
        | StrOutputParser()                      # get plain string instead of AIMessage
    )

    text = rag_chain.invoke(question)            # returns str thanks to StrOutputParser
    return {"answer": text}
