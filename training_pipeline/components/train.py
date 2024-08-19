"""Creates and runs training component in kfp pipeline"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from keras.utils import to_categorical  # type: ignore
from kfp.dsl import ContainerSpec, container_component
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from training_pipeline.components.utils import (
    compute_metrics,
    get_label_encoder,
    load_data,
    preprocess_dataset,
    upload_blob,
)
from training_pipeline.config import (
    DATALOADER_NAME,
    FILE_BUCKET,
    IMAGE_TRAIN_LOC,
    MODEL_NAME,
)


@container_component
def train_model_component() -> ContainerSpec:
    return ContainerSpec(
        image=IMAGE_TRAIN_LOC,
        command=["python", "-m", "training_pipeline.components.train"],
    )


def train_model() -> None:
    """Train a predictive model to classify news articles into categories."""
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get data
    df = load_data()

    # Give df as argument if LabelEncoder does not exist or is out of date
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

    print("torch device: ", device)
    model.to(device)

    model.train()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        all_preds = []
        all_labels = []
        for i, batch in enumerate(train_dataloader):
            # if i > 100: break
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

    temp_dir = tempfile.mkdtemp()
    model_path = f"{Path(temp_dir)}/{MODEL_NAME}"
    dataloader_path = f"{Path(temp_dir)}/{DATALOADER_NAME}"

    torch.save(model.state_dict(), model_path)
    upload_blob(FILE_BUCKET, model_path, MODEL_NAME)

    torch.save(test_dataloader, dataloader_path)
    upload_blob(FILE_BUCKET, dataloader_path, DATALOADER_NAME)


if __name__ == "__main__":
    train_model()
