resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-fish-id-models"
  location                    = var.region
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis["storage.googleapis.com"]]
}

# CI/CD SA can download the model during container builds
resource "google_storage_bucket_iam_member" "cicd_model_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cicd.email}"
}
