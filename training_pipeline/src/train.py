import torch
import numpy as np
from torch.utils.data import DataLoader
from kfp.v2.dsl import component
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)

from training_pipeline.config import DATASET_LOC, MODEL_LOC
from training_pipeline.src.utils import (
    compute_metrics,
    load_data,
    preprocess_dataset,
)
from keras.utils import to_categorical  # type: ignore


@component(
    base_image="python:3.10",
    output_component_file="train_model.yaml",
)
def train_model(label_encoder):
    """Train a predictive model to classify news articles into categories."""
    # Get data
    df = load_data(DATASET_LOC)

    # Give df as argument if LabelEncoder does not exist or is out of date
    # label_encoder = get_label_encoder(df)

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

    model.train()

    # Training loop
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

    torch.save(model.state_dict(), MODEL_LOC)
    return (model, test_dataloader)
