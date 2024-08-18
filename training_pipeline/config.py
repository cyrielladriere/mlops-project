"""Module for training_pipeline wide configurations."""

MODEL_LOCATION = "full_model.pt"
DATA_LOCATION = "training_pipeline/data/news_data.json"

PIPELINE_NAME = "news-articles-pipeline"

PROJECT_ID = "mlops-project-431716"
REGION = "europe-west1"

FILE_BUCKET = "mlops_project_data"
DATASET_ID = "news_articles"

DATASET_NAME = f"{DATASET_ID}_data.csv"
LABEL_ENCODER_NAME = f"{DATASET_ID}_label_encoder.pkl"
MODEL_NAME = f"{DATASET_ID}_model.pt"
DATALOADER_NAME = f"{DATASET_ID}_test_dataloader.pt"

VERTEX_SVC = f"svc-vertex@{PROJECT_ID}.iam.gserviceaccount.com"

IMAGE_LOC = f"europe-west1-docker.pkg.dev/\
    {PROJECT_ID}/mlops-project-image-repository/\
    mlops-image:latest"
