FROM python:3.10-slim

WORKDIR /app

COPY ./api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

COPY ./api/main.py /app/api/main.py
COPY ./src/preprocess.py /app/src/preprocess.py

COPY ./mlruns /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

