FROM python:3.10-slim

WORKDIR /usr/src/app

COPY training_pipeline/src/ ./src/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install Torch with GPU
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118