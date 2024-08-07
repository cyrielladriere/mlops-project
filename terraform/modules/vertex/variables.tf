variable "project" {
  description = "GCP project name"
  type        = string
}

variable "svc_vertex_id" {
  description = "Service account id. {account_id}@{project_id}.iam.gserviceaccount.com"
  type        = string
}

variable "svc_vertex_roles" {
  description = "IAM roles to bind on service account"
  type        = list(string)
}

variable "bucket_name" {
  description = "Name of the bucket to create"
  type        = string
}
