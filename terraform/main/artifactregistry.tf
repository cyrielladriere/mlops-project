# resource "google_project_service" "artifactregistry" {
#   project = var.project
#   service = "artifactregistry.googleapis.com"
# }

# resource "google_artifact_registry_repository" "docker_artifact" {
#   location      = var.region
#   project       = var.project
#   repository_id = "mlops-project-image-repository"
#   format        = "DOCKER"
# }
