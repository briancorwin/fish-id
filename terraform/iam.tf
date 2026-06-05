data "google_project" "project" {}

# Cloud Run service account — no IAM roles; the app makes no GCP API calls
resource "google_service_account" "cloud_run" {
  account_id   = "fish-id-cloud-run-sa"
  display_name = "fish-id Cloud Run SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# CI/CD service account — impersonated by GitHub Actions via Workload Identity
resource "google_service_account" "cicd" {
  account_id   = "fish-id-cicd-sa"
  display_name = "fish-id CI/CD SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Workload Identity pool — one pool per project is enough
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis["iamcredentials.googleapis.com"]]
}

# GitHub OIDC provider — restricts to a single repo via attribute condition
resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  attribute_condition = "attribute.repository == \"${var.github_repo}\" && attribute.ref == \"refs/heads/main\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Allow GitHub Actions workflows in the repo to impersonate the CI/CD SA
resource "google_service_account_iam_member" "cicd_wif" {
  service_account_id = google_service_account.cicd.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# Deploy and update Cloud Run services
resource "google_project_iam_member" "cicd_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

# Attach the Cloud Run SA when deploying
resource "google_service_account_iam_member" "cicd_sa_user" {
  service_account_id = google_service_account.cloud_run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd.email}"
}

# Push container images to Artifact Registry
resource "google_artifact_registry_repository_iam_member" "cicd_ar_writer" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.fish_id.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.cicd.email}"
}

# Deploy Firebase Hosting
resource "google_project_iam_member" "cicd_firebase_hosting" {
  project = var.project_id
  role    = "roles/firebasehosting.admin"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

# CI/CD SA — read and write model artifacts (download for deploy, write training-image-latest.json)
resource "google_storage_bucket_iam_member" "cicd_model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cicd.email}"
}

# Training service account — used by Vertex AI CustomJob containers
resource "google_service_account" "training" {
  account_id   = "fish-id-training-sa"
  display_name = "fish-id Training SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Training SA — read training images and labels
resource "google_storage_bucket_iam_member" "training_training_bucket_reader" {
  bucket = google_storage_bucket.training.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.training.email}"
}

# Training SA — write trained model artifacts to the models bucket
resource "google_storage_bucket_iam_member" "training_model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.training.email}"
}

# Training SA — submit Vertex AI custom jobs
resource "google_project_iam_member" "training_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.training.email}"
}

# Workflows service account — used by KFP pipeline components
resource "google_service_account" "workflows" {
  account_id   = "fish-id-workflows-sa"
  display_name = "fish-id Workflows SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Workflows SA — pipeline root artifacts and model bucket reads
resource "google_storage_bucket_iam_member" "workflows_model_admin" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — submit Vertex AI pipeline and custom jobs
resource "google_project_iam_member" "workflows_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — write logs
resource "google_project_iam_member" "workflows_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — act as the training SA when submitting Vertex AI CustomJobs
resource "google_service_account_iam_member" "workflows_act_as_training" {
  service_account_id = google_service_account.training.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workflows.email}"
}
