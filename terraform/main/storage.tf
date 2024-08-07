module "storage" {
  source      = "../modules/storage"
  project     = var.project
  bucket_name = "mlops_project_data"
}

variable "svc_storage_id" {
  description = "Service account id. {account_id}@{project_id}.iam.gserviceaccount.com"
  type        = string
  default     = "svc-storage"
}

variable "svc_storage_roles" {
  description = "IAM roles to bind on service account"
  type        = list(string)
  default = [
    "roles/editor"
  ]
}
