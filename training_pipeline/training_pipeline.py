"""Module for the news articles pipeline definition."""
from kfp.dsl import pipeline

from training_pipeline import config
from training_pipeline.components.eval import eval_model_component
from training_pipeline.components.train import train_model_component
from training_pipeline.components.upload_dataset import upload_data_component


@pipeline(name=config.PIPELINE_NAME, description=config.PIPELINE_NAME)
def pipeline() -> None:
    """
    Define a Kubeflow pipeline for processing news articles and
    training/evaluating a BERT model on that data.
    """
    import_data_op = upload_data_component()

    model_training_op = train_model_component()

    model_evaluation_op = eval_model_component()

    model_training_op.after(import_data_op)
    model_evaluation_op.after(model_training_op)
