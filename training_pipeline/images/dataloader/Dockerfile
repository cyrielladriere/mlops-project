FROM python:3.10-slim

WORKDIR /usr/src/app

COPY training_pipeline/images/dataloader/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY training_pipeline/ ./training_pipeline/