from kfp.v2.dsl import component

from training_pipeline.config import DATA_LOCATION
from training_pipeline.src.utils import load_data


@component(
    base_image="python:3.10",
    output_component_file="upload_data.yaml",
)
def upload_data(
    dataset_id: str,
    file_bucket: str,
) -> str:
    df = load_data(DATA_LOCATION)

    # save df in cloud storage
    save_path = f"gs://{file_bucket}/{dataset_id}/{dataset_id}_data.csv"
    df.to_csv(save_path, index=True)

    print(f"{dataset_id}_data.csv saved in {save_path}")

    return save_path
