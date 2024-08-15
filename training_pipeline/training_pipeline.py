from kfp.v2.dsl import pipeline
from training_pipeline import config
from training_pipeline.src.eval import eval_model
from training_pipeline.src.train import train_model
from training_pipeline.src.upload_dataset import upload_data


@pipeline(name=config.PIPELINE_NAME, description=config.PIPELINE_NAME)
def pipeline(name: str = config.PIPELINE_NAME):
    import_data_op = upload_data()

    model_training_op = train_model()

    model_evaluation_op = eval_model()

    model_training_op.after(import_data_op)
    model_evaluation_op.after(model_training_op)
