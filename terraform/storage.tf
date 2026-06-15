# Terraform remote state bucket. Must be created manually before terraform init —
# see the bootstrap instructions in main.tf. Import after first apply:
#   terraform import google_storage_bucket.terraform_state PROJECT_ID-fish-id-terraform-state
resource "google_storage_bucket" "terraform_state" {
  name                        = "${var.project_id}-fish-id-terraform-state"
  location                    = var.region
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis["storage.googleapis.com"]]
}

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

# Training data bucket — flat pool of images and labels, versioned so each sync is recoverable
resource "google_storage_bucket" "training" {
  name                        = "${var.project_id}-fish-id-training"
  location                    = var.region
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition { num_newer_versions = 10 }
    action    { type = "Delete" }
  }

  depends_on = [google_project_service.apis["storage.googleapis.com"]]
}
