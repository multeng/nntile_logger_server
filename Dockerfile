FROM python:3.11-slim

RUN apt-get update

COPY . ./app

WORKDIR /app


CMD ["python", "-u", "server.py"]
