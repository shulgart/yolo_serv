# syntax=docker/dockerfile:1
FROM python:3.12.2-slim

RUN pip install pipenv

WORKDIR /app

# libGL.so.1 problem solved
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# COPY ["object_detector.py", "index.html", "best.pt", "/images"]

# copy source code into working directory
COPY . /app

EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=8080", "object_detector:app"]