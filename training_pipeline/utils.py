"""Helper functions for the MLOps Project."""
from typing import Any, Dict, List
import pickle  # nosec

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

LABEL_ENCODER_LOC = "training_pipeline/data/label_encoder.pkl"


def load_data(data_file: str) -> pd.DataFrame:
    """Load and preprocess the news article data from a JSON file."""
    df = pd.read_json(data_file, lines=True)

    return df[["category", "short_description"]]


def get_label_encoder(df: pd.DataFrame | None = None):
    if df is None:
        with open(LABEL_ENCODER_LOC, "rb") as file:
            return pickle.load(file)  # nosec

    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"])

    with open(LABEL_ENCODER_LOC, "wb") as file:
        pickle.dump(label_encoder, file)

    return label_encoder


def preprocess_dataset(data: pd.DataFrame, tokenizer) -> Dataset:
    """Preprocess and tokenize the dataset using a BERT tokenizer."""
    dataset = Dataset.from_pandas(data)

    def tokenize_dataset(examples: Dict[str, List[str]]) -> Any:
        return tokenizer(
            examples["short_description"],
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
