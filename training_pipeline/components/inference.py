"""Inference implementation of news articles model"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)

from training_pipeline.config import MODEL_LOCATION
from training_pipeline.components.utils import (
    get_label_encoder,
    load_data,
    preprocess_dataset,
)


def predict(input_json):
    """Predict news categories based on news articles using a pre-trained model."""
    data = load_data(input_json)

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


def predict_categories(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device,
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
