"""Creates and runs upload_data component in kfp pipeline"""
import tempfile
from pathlib import Path

import pandas as pd
from kfp.dsl import ContainerSpec, container_component

from training_pipeline.components.utils import get_label_encoder, upload_blob
from training_pipeline.config import (
    DATA_LOCATION,
    DATASET_NAME,
    FILE_BUCKET,
    IMAGE_DATALOADER_LOC,
)


@container_component
def upload_data_component():
    return ContainerSpec(
        image=IMAGE_DATALOADER_LOC,
        command=["python", "-m", "training_pipeline.components.upload_dataset"],
    )


def upload_data() -> None:
    df = pd.read_json(DATA_LOCATION, lines=True)

    get_label_encoder(df)

    temp_dir = tempfile.mkdtemp()
    temp_path = f"{Path(temp_dir)}/{DATASET_NAME}"
    df.to_csv(temp_path)

    upload_blob(FILE_BUCKET, temp_path, DATASET_NAME)


if __name__ == "__main__":
    upload_data()
