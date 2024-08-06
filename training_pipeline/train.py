from io import BytesIO
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import (BertForSequenceClassification,
                          get_linear_schedule_with_warmup, BertTokenizer)
from datasets import load_from_disk

from training_pipeline.helper import (eval_model, get_label_encoder, load_data, eval_model, predict_categories, preprocess_dataset,
                              train_model)
from keras.utils import to_categorical 

DATA_LOCATION = "training_pipeline/data/news_data.json"
MODEL_LOCATION = "full_model.pt"

def train() -> None:
    """Train a predictive model to rank news categories based on their news article description."""
    # Get data
    df = load_data(DATA_LOCATION)

    # Give df as argument if LabelEncoder does not exist or is out of date
    # label_encoder = get_label_encoder(df)
    label_encoder = get_label_encoder()
    df["label"] = label_encoder.transform(df["category"])
    n_classes = len(label_encoder.classes_)
    df["label"] = df["label"].apply(to_categorical, num_classes=n_classes)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Preprocess and tokenize dataset + train-test split
    datasets = preprocess_dataset(df, tokenizer).train_test_split(test_size=0.3)
    
    # datasets.save_to_disk("temp_dataset")
    # datasets = load_from_disk("temp_dataset.hf")

    # DataLoader
    train_dataloader = DataLoader(datasets["train"], shuffle=False, batch_size=8)
    test_dataloader = DataLoader(datasets["test"], shuffle=False, batch_size=8) 

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=n_classes,
    )

    # Optimizer and scheduler
    num_epochs = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(datasets["train"]) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs)

    preds = eval_model(model, test_dataloader, label_encoder, device)
    datasets["test"] = datasets["test"].add_column("predictions", preds)
    


def predict() -> Dict[int, Dict[int, str]]:
    """Predict news categories based on their news article short description using a pre-trained model."""
    data = load_data(DATA_LOCATION)[:1000]

    label_encoder = get_label_encoder()
    n_classes = len(label_encoder.classes_)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=n_classes
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(MODEL_LOCATION))
    model.to(device)

    # Preprocess and tokenize dataset
    dataset = preprocess_dataset(data, tokenizer)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=8)

    results = predict_categories(model, dataloader, label_encoder, device)

    # # Append results to original dataset
    # dataset = dataset.add_column("predictions", results)

    return results

train()