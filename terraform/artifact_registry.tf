resource "google_artifact_registry_repository" "fish_id" {
  repository_id = "fish-id"
  format        = "DOCKER"
  location      = var.region
  description   = "Container images for fish-id Cloud Run service"

  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}
