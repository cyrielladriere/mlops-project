resource "google_storage_bucket" "data-bucket" {
  name     = var.bucket_name
  project  = var.project
  location = "europe-west1"

  versioning {
    enabled = true
  }
}
