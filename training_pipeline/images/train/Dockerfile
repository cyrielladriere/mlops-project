FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest

WORKDIR /usr/src/app

COPY training_pipeline/images/train/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY training_pipeline/ ./training_pipeline/

# Reset entrypoint by parent image
ENTRYPOINT []