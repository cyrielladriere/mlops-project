from kfp.v2.dsl import component

from training_pipeline.config import DATA_LOCATION, DATASET_LOC, LABEL_ENCODER_LOC
from training_pipeline.src.utils import get_label_encoder, load_data


@component(
    base_image="python:3.10",
    output_component_file="upload_data.yaml",
)
def upload_data(
    dataset_id: str,
    file_bucket: str,
) -> str:
    df = load_data(DATA_LOCATION)
    label_encoder = get_label_encoder(df)

    df.to_csv(DATASET_LOC, index=True)

    print(f"{dataset_id}_data.csv saved in {DATA_LOCATION}")
    print(f"Label encoder saved in {LABEL_ENCODER_LOC}")

    return label_encoder
