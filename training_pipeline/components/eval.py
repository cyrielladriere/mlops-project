"""Creates and runs the evaluation component in kfp pipeline"""
import io

import numpy as np
import torch
from kfp.dsl import ContainerSpec, container_component
from transformers import BertForSequenceClassification

from training_pipeline.components.utils import (
    compute_metrics,
    get_label_encoder,
    read_blob,
)
from training_pipeline.config import (
    DATALOADER_NAME,
    FILE_BUCKET,
    IMAGE_TRAIN_LOC,
    MODEL_NAME,
)


@container_component
def eval_model_component():
    return ContainerSpec(
        image=IMAGE_TRAIN_LOC,
        command=["python", "-m", "training_pipeline.components.eval"],
    )


def eval_model() -> None:
    """Predict news category for news articles using the trained model."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataloader_bytes = read_blob(FILE_BUCKET, DATALOADER_NAME)
    dataloader = torch.load(io.BytesIO(dataloader_bytes))

    label_encoder = get_label_encoder()
    n_classes = len(label_encoder.classes_)

    model_bytes = read_blob(FILE_BUCKET, MODEL_NAME)
    model_dict = torch.load(io.BytesIO(model_bytes))

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=n_classes
    )
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    for i, batch in enumerate(dataloader):
        # if i > 100: break
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


if __name__ == "__main__":
    eval_model()
