module "vertex" {
  source           = "../modules/vertex"
  project          = var.project
  bucket_name      = "${var.project}_vertex-pipelines"
  svc_vertex_id    = var.svc_vertex_id
  svc_vertex_roles = var.svc_vertex_roles
}

variable "svc_vertex_id" {
  description = "Service account id. {account_id}@{project_id}.iam.gserviceaccount.com"
  type        = string
  default     = "svc-vertex"
}

variable "svc_vertex_roles" {
  description = "IAM roles to bind on service account"
  type        = list(string)
  default = [
    "roles/storage.objectUser",
    "roles/aiplatform.admin",
  ]
}
