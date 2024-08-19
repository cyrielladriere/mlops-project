"""Deploys and runs the kfp news articles pipeline."""
import logging
import subprocess  # nosec
import sys
import tempfile
import warnings
from pathlib import Path

from google.cloud import aiplatform
from kfp.compiler import Compiler

from training_pipeline import config
from training_pipeline.training_pipeline import pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def deploy_pipeline() -> None:
    """Deploys the Kubeflow pipeline to Google Cloud Vertex AI."""
    logger.info("Initializing Vertex AI")
    aiplatform.init(
        project=config.PROJECT_ID,
        location=config.REGION,
        service_account=config.VERTEX_SVC,
    )

    temp_dir = tempfile.mkdtemp()
    temp_path = f"{Path(temp_dir)}/pipeline.yaml"

    logger.info("Compiling Kubeflow Pipeline")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Compiler().compile(pipeline_func=pipeline, package_path=temp_path)

    logger.info("Building and pushing Docker images to registry")
    subprocess.run(["./training_pipeline/images/build_and_push_all.sh"])  # nosec

    logger.info("Starting Pipeline Job")
    job = aiplatform.PipelineJob(
        display_name=config.PIPELINE_NAME,
        template_path=temp_path,
        enable_caching=False,  # True just for testing, to save resources
        # pipeline_root=f"gs://{config.FILE_BUCKET}",
    )

    job.run()


if __name__ == "__main__":
    deploy_pipeline()
