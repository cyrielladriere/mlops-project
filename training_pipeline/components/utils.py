"""Helper functions for the MLOps Project."""
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import pickle  # nosec

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from google.cloud import storage

from training_pipeline.config import FILE_BUCKET, LABEL_ENCODER_NAME, DATASET_NAME


def load_data() -> pd.DataFrame:
    """Load and preprocess the news article data from a GCP bucket."""
    df = pd.read_csv(f"gs://{FILE_BUCKET}/{DATASET_NAME}")

    return df[["category", "short_description"]]


def get_label_encoder(df: pd.DataFrame | None = None):
    if df is None:
        bytes = read_blob(FILE_BUCKET, LABEL_ENCODER_NAME)
        return pickle.loads(bytes)  # nosec

    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"])

    temp_dir = tempfile.mkdtemp()
    temp_path = f"{Path(temp_dir)}/label_encoder.pkl"

    with open(temp_path, "wb+") as file:
        pickle.dump(label_encoder, file)

    upload_blob(FILE_BUCKET, temp_path, LABEL_ENCODER_NAME)

    return label_encoder


def upload_blob(
    bucket_name: str, source_file_path: str, destination_blob_name: str
) -> None:
    """
    Uploads a file to a specified bucket in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the bucket to upload the file to.
        source_file_path (str): The local file path of the file to upload.
        destination_blob_name (str): The name of the blob (file) in the bucket.

    Returns:
        None
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path, timeout=10 * 60)

    print(
        f"File {source_file_path} uploaded to \
        gs://{bucket_name}/{destination_blob_name}."
    )


def read_blob(bucket_name: str, file_name: str, text: bool = False) -> bytes:
    """
    Reads a file from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCP bucket.
        file_name (str): The name of the file in the bucket.

    Returns:
        bytes: The contents of the file as bytes.
    """
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(file_name)

    if text:
        return blob.download_as_text()
    else:
        return blob.download_as_bytes()


def preprocess_dataset(data: pd.DataFrame, tokenizer) -> Dataset:
    """Preprocess and tokenize the dataset using a BERT tokenizer."""
    # FIX: For some reason after downloading csv from gcp bucket we have
    # ~2000 nan values in the "short desciption" column
    # (nans were not present before upload to gcp)
    data = data.dropna()
    dataset = Dataset.from_pandas(data)

    def tokenize_dataset(examples: Dict[str, List[str]]) -> Any:
        return tokenizer(
            text_target=examples["short_description"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset = dataset.remove_columns(["short_description", "category"])
    dataset = dataset.with_format("torch")

    return dataset


def compute_metrics(pred, labels):
    labels = labels.argmax(-1)
    preds = pred.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
