resource "google_service_account" "vertex" {
  project     = var.project
  account_id  = var.svc_vertex_id
  description = "Service account to run Vertex AI pipelines"
}

resource "google_project_iam_member" "vertex" {
  for_each = toset(var.svc_vertex_roles)
  role     = each.key
  project  = var.project
  member   = "serviceAccount:${google_service_account.vertex.email}"
}

# resource "google_storage_bucket" "vertex-bucket" {
#   name     = var.bucket_name
#   project  = var.project
#   location = "europe-west1"

#   versioning {
#     enabled = true
#   }
# }
