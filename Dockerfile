FROM pytorch/pytorch:1.8.1-cuda11.0-cudnn8-runtime

WORKDIR /code

COPY requirements.txt .
COPY src/models/ ./src/models/
COPY setup.py .

RUN pip3 install -r requirements.txt

WORKDIR /
