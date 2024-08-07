import numpy as np
import torch
from kfp.v2.dsl import component
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from training_pipeline.src.utils import compute_metrics, get_label_encoder


@component(
    base_image="python:3.10",
    output_component_file="eval_model.yaml",
)
def eval_model(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
) -> np.ndarray:
    """Predict news category for news articles using the trained model."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_encoder = get_label_encoder()

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
