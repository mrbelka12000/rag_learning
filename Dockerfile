FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install "fastapi[standard]" openai-whisper langchain

COPY *.py /code

EXPOSE 8080

CMD ["fastapi", "run", "main.py", "--port", "8080", "--host", "0.0.0.0"]