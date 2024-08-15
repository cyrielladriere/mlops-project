from kfp.v2.dsl import component

from training_pipeline.config import (
    DATA_LOCATION,
    DATASET_ID,
    DATASET_LOC,
    LABEL_ENCODER_LOC,
)
from training_pipeline.src.utils import get_label_encoder, load_data


@component(
    base_image="python:3.10",
    output_component_file="upload_data.yaml",
)
def upload_data() -> None:
    df = load_data(DATA_LOCATION)
    get_label_encoder(df)

    df.to_csv(DATASET_LOC, index=True)

    print(f"{DATASET_ID}_data.csv saved in {DATA_LOCATION}")
    print(f"Label encoder saved in {LABEL_ENCODER_LOC}")
