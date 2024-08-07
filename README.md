# mlops-project
!! This repository is a heavy Work In Progress!!

Our task: fine-tuning BERT to classify news articles into news categories. The plan is to expand this project to an mlops project. Using technologies such as: Terraform, Docker, KubeFlow Pipelines, cloud architecture (gcp).

Current performance full_model on train set: /

Current performance full_model on test set: /

Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download

# Terraform

The first step of setting up Terraform, is to initialise terraform in the `terraform/main` folder:
```sh
terraform init
```
Afterwards, we can use `terraform plan` to see what resources are going to be created/changed:
```sh
terraform plan -var-file="../environment/project.tfvars"
```