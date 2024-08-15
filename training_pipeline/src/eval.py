import numpy as np
import torch
from kfp.v2.dsl import component
from transformers import BertForSequenceClassification

from training_pipeline.config import DATALOADER_LOC, MODEL_LOC
from training_pipeline.src.utils import compute_metrics, get_label_encoder


@component(
    base_image="python:3.10",
    output_component_file="eval_model.yaml",
)
def eval_model() -> None:
    """Predict news category for news articles using the trained model."""
    dataloader = torch.load(DATALOADER_LOC)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    label_encoder = get_label_encoder()
    n_classes = len(label_encoder.classes_)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=n_classes
    )
    model.load_state_dict(torch.load(MODEL_LOC))

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
    print(label_encoder.inverse_transform(all_preds))
