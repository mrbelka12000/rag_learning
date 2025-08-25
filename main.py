import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from vectors import make_vectors
app = FastAPI()
app.state.retriever = None  # will be set on startup

default_path = "data"

if __name__ == "__main__":
    load_dotenv()
    print("This is the main module.")
    run(default_path)
    print("Vector store is ready.")


from pydantic import BaseModel
class AnswerRequest(BaseModel):
    question: str

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")


@app.on_event("startup")
def startup():
    from core import run
    load_dotenv()
    vs = run(default_path)
    app.state.retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})


@app.post("/answer")
async def endpoint(req: AnswerRequest):
    from answer import answer
    result = answer(req.question, app.state.retriever)
    # return proper JSON (dict), not an f-string
    return result