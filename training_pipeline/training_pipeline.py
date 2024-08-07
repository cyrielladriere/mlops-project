from kfp.v2.dsl import pipeline
from training_pipeline import config
from training_pipeline.src.eval import eval_model
from training_pipeline.src.train import train_model
from training_pipeline.src.upload_dataset import upload_data


@pipeline(name="news-articles-pipeline", description="news-articles-pipeline")
def pipeline(
    name=config.PIPELINE_NAME,
):
    import_data_op = upload_data()

    model_training_op = train_model(import_data_op.output)

    model_evaluation_op = eval_model(
        model_training_op.outputs["model"], model_training_op.outputs["test_dataloader"]
    )

    model_training_op.after(import_data_op)
    model_evaluation_op.after(model_training_op)
