FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

WORKDIR /usr/src/app

COPY training_pipeline/images/train/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY training_pipeline/ ./training_pipeline/