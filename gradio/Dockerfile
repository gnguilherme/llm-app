FROM python:3.11-slim

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./app.py /app/app.py
RUN pip install -r requirements.txt
