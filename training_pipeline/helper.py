"""Helper functions for the MLOps Project."""
from io import BytesIO
from typing import Any, Dict, List
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

LABEL_ENCODER_LOC = "training_pipeline/data/label_encoder.pkl"

def train_model(
    model: BertForSequenceClassification,
    train_dataloader: DataLoader[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int
) -> None:
    """Train the BERT model for news category classification."""
    model.train()

    for epoch in range(1, num_epochs + 1):
        all_preds = []
        all_labels = []
        for i, batch in enumerate(train_dataloader):
            print(f"[Epoch: {epoch}] batch: {i}/{len(train_dataloader)}")
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"].float(),
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = outputs.logits.detach().cpu().numpy()
            labels = batch["label"].float().detach().cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_metrics(all_preds, all_labels)
        print("Performance model on train set: ", metrics)

    torch.save(model.state_dict(), "./model.pt")


def eval_model(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device
) -> np.ndarray:
    """Predict news category for news articles using the trained model."""
    model.eval()
    all_preds = []
    all_labels = []
    for i, batch in enumerate(dataloader):
        print(f"[Eval] batch: {i}/{len(dataloader)}")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

        preds = outputs.logits.detach().cpu().numpy()
        all_preds.append(preds)

        labels = batch["label"].float().detach().cpu().numpy()
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_preds, all_labels)
    print("Performance model on test set: ", metrics)

    all_preds = np.argmax(all_preds, axis=1)
    return label_encoder.inverse_transform(all_preds)


def predict_categories(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device
) -> np.ndarray:
    """Predict news category for news articles using the trained model."""
    model.eval()
    all_preds = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

        preds = outputs.logits.detach().cpu().numpy()
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = np.argmax(all_preds, axis=1)

    return label_encoder.inverse_transform(all_preds)


def load_data(data_file: str) -> pd.DataFrame:
    """Load and preprocess the news article data from a JSON file."""
    df = pd.read_json(data_file, lines=True)

    return df[["category", "short_description"]]

def get_label_encoder(df: pd.DataFrame | None = None):
    if df is None:
        with open(LABEL_ENCODER_LOC, 'rb') as file:
            return pickle.load(file)
        
    label_encoder = LabelEncoder()
    label_encoder.fit(df["category"])

    with open(LABEL_ENCODER_LOC, 'wb') as file:
        pickle.dump(label_encoder, file)
    
    return label_encoder



def preprocess_dataset(data: pd.DataFrame, tokenizer) -> Dataset:
    """Preprocess and tokenize the dataset using a BERT tokenizer."""
    dataset = Dataset.from_pandas(data)

    def tokenize_dataset(examples: Dict[str, List[str]]) -> Any:
        return tokenizer(
            examples["short_description"], padding="max_length", truncation=True, max_length=512
        )

    dataset = dataset.map(tokenize_dataset, batched=True)
    dataset = dataset.remove_columns(["short_description", "category"])
    dataset = dataset.with_format("torch")

    return dataset

def compute_metrics(pred, labels):
    labels = labels.argmax(-1)
    preds = pred.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }