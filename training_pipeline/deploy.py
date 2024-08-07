from pathlib import Path
import tempfile
import warnings
from google.cloud import aiplatform
from kfp.compiler import Compiler
from training_pipeline import config
from training_pipeline.training_pipeline import pipeline


def deploy_pipeline():
    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)

    temp_dir = tempfile.mkdtemp()
    temp_path = f"{Path(temp_dir)}/pipeline.yaml"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Compiler().compile(pipeline_func=pipeline, package_path=temp_path)

    job = aiplatform.PipelineJob(
        display_name=config.PIPELINE_NAME,
        template_path=temp_path,
        parameter_values={"project_id": config.PROJECT_ID},
        enable_caching=True,  # True just for testing, to save resources
        pipeline_root=f"gs://{config.FILE_BUCKET}",
    )

    job.run()
