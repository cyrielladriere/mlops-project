from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)

from training_pipeline.utils import (
    compute_metrics,
    get_label_encoder,
    load_data,
    preprocess_dataset,
)
from keras.utils import to_categorical

DATA_LOCATION = "training_pipeline/data/news_data.json"
MODEL_LOCATION = "full_model.pt"


def train() -> None:
    """Train a predictive model to classify news articles into categories."""
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
    DataLoader(datasets["test"], shuffle=False, batch_size=8)

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

    training_loop(model, train_dataloader, optimizer, scheduler, device, num_epochs)

    # preds = eval_model(model, test_dataloader, label_encoder, device)
    # datasets["test"] = datasets["test"].add_column("predictions", preds)


def training_loop(
    model: BertForSequenceClassification,
    train_dataloader: DataLoader[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
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


train()
