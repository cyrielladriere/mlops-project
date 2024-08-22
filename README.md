# mlops-project
This is an end-to-end MLOps project for training and evaluating a machine learning model using cloud resources (Google Cloud Platform).

!! This repository is a heavy Work In Progress !!

# Problem
The problem itself is simple: fine-tuning a BERT transformer architecture to classify news articles into news categories. 

The pre-trained BERT model is imported from huggingface. Data and models are stored in gcp storage buckets. 

Kubeflow pipelines is used to create the machine learning pipeline in Vertex AI. This pipeline consists of multiple components (more info: [`training_pipeline/README.md`](./training_pipeline/README.md)), each with their own docker container image. These images are built and pushed to the Google Cloud Artifact Registry using the `training_pipeline/images/build_and_push_all.sh` script. 

All cloud resources and service accounts are created using Terraform (see `terraform` folder).

Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download

# Setup
This repository was developed in python 3.10. Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```

# Pre-commit
Pre-commit is used to enable a smooth developer flow. Run the following commands to set up your development environment:
```sh
pre-commit install
```

# Terraform
The first step of setting up Terraform, is to initialise terraform in the `terraform/main` folder:
```sh
terraform init
```
Afterwards, we can use `terraform plan` to see what resources are going to be created/changed:
```sh
terraform plan -var-file="../environment/project.tfvars"
```
When we actually want to apply the changes we can execute `terraform apply`:
```sh
terraform apply -var-file="../environment/project.tfvars"
```

# Building and pushing Docker images
First, we have to set up authentication:
```sh
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

Once you are authenticated, run the `build_and_push_all.sh` bash script:
```sh
./training_pipeline/images/build_and_push_all.sh 
```
This script will build, tag, and push the necessary images to the google cloud artifact registry inside the mlops GCP project. 

A tip: I enabled image vulnerabilty scanning in the gcp registry by accident and got stuck with a sizeable bill, so do not enable this if it is not needed!

**Note:** The created images are based on the Dockerfiles in `training_pipeline/images/{image_name}/Dockerfile`

# Running Kubeflow Pipeline in Vertex AI
The Kubeflow Pipeline can be executed by running the following command:
```sh
python -m training_pipeline.deploy
```

More info on the pipeline in [`training_pipeline/README.md`](training_pipeline/README.md).

# TODO's
- FIX: For some reason after downloading csv from gcp bucket we have ~2000 nan values in the "short desciption" column (nans were not present before upload to gcp), see `preprocess_dataset` function in `training_pipeline/components/utils.py`.(temp fix implemented: remove nans after pulling from bucket)
- Expand pipeline with extra functionality
- Add CI/CD pipeline with github actions.
- Add testing (and add it in CI/CD pipeline)
- Transfer deployment to a CI/CD job running in a container.
- Add model experiment tracking (mlflow, tensorboard, ...)

