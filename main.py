from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
load_dotenv()

app = FastAPI()
app.state.retriever = None  # will be set on startup

default_path = "data"

if __name__ == "__main__":
    print("This is the main module.")

from pydantic import BaseModel

class AnswerRequest(BaseModel):
    question: str

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

@app.on_event("startup")
def startup():
    import shutil
    from core import run_vectorization
    load_dotenv()
    vs = run_vectorization(default_path)
    app.state.retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    print("Vector store is ready.")
    shutil.rmtree(default_path)

@app.post("/answer")
async def endpoint(req: AnswerRequest):
    from answer import answer
    result = answer(req.question, app.state.retriever)
    return result
