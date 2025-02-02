"""Creates and runs upload_data component in kfp pipeline."""
import logging
import sys
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@container_component
def upload_data_component() -> ContainerSpec:
    """
    Define a Kubeflow container component for uploading data
    to a GCP storage bucket.
    """
    return ContainerSpec(
        image=IMAGE_DATALOADER_LOC,
        command=["python", "-m", "training_pipeline.components.upload_dataset"],
    )


def upload_data() -> None:
    """Upload data and a fitted label encoder to GCP storage bucket."""
    logger.info("Reading dataset")
    df = pd.read_json(DATA_LOCATION, lines=True)

    logger.info("Fitting and storing label encoder")
    get_label_encoder(df)

    temp_dir = tempfile.mkdtemp()
    temp_path = f"{Path(temp_dir)}/{DATASET_NAME}"
    df.to_csv(temp_path)

    logger.info("Uploading dataset as csv to gcp bucket")
    upload_blob(FILE_BUCKET, temp_path, DATASET_NAME)


if __name__ == "__main__":
    upload_data()
