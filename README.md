# rag_learning


## RAG stands for Retrieval-Augmented Generation is an AI framework that enhances Large Language Models (LLMs) by first retrieving relevant information from external sources and then using that information to generate more accurate, context-specific, and up-to-date responses.


## Tech stack:
1. langchain - For generating embeddings from provided data
2. chromadb - For local vector database
3. fastapi - For backend server
4. python-dotenv - For loading environmental variables
5. pypdf - For reading pdf files



## How to run program:
1. Create ".env" file in the root of project and add OPENAI_API_KEY=sk-proj-...
2. Create directory named "data" in the root of project.
3. Provide some files from which LLM will retrieve data and embeddings will construct.
4. Run commands
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```
4. Go to https://localhost:8000/docs to check it out.
5. Fell free to ask any information from provided documents using "/answer" endpoint.
