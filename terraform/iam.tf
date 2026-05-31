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

# Deploy and update Cloud Run services — developer role excludes service deletion and IAM policy management
resource "google_project_iam_member" "cicd_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

# Attach the Cloud Run SA when deploying (required by gcloud run deploy --service-account)
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

# Read the ONNX model from GCS (bucket-level binding in storage.tf)

# CI/CD SA — write new versioned models to the models bucket
resource "google_storage_bucket_iam_member" "cicd_model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.cicd.email}"
}

# Training service account — used by Vertex AI CustomJob containers
resource "google_service_account" "training" {
  account_id   = "fish-id-training-sa"
  display_name = "fish-id Training SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Training SA — full object access on the training data bucket
resource "google_storage_bucket_iam_member" "training_training_bucket_admin" {
  bucket = google_storage_bucket.training.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.training.email}"
}

# Training SA — write trained model artifacts to the models bucket
resource "google_storage_bucket_iam_member" "training_model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.training.email}"
}

# Training SA — submit and manage Vertex AI custom jobs
resource "google_project_iam_member" "training_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.training.email}"
}

# Workflows service account — used by Cloud Workflows execution
resource "google_service_account" "workflows" {
  account_id   = "fish-id-workflows-sa"
  display_name = "fish-id Workflows SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Workflows SA — read model artifacts from the models bucket
resource "google_storage_bucket_iam_member" "workflows_model_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — write model artifacts to the models bucket
resource "google_storage_bucket_iam_member" "workflows_model_writer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — submit and manage Vertex AI custom jobs
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

# Workflows SA — access the GitHub PAT secret
resource "google_secret_manager_secret_iam_member" "workflows_github_pat_accessor" {
  secret_id = google_secret_manager_secret.github_deploy_pat.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — receive Eventarc events (required for Eventarc trigger service account)
resource "google_project_iam_member" "workflows_eventarc_receiver" {
  project = var.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — invoke Cloud Workflows executions (required for Eventarc to start the workflow)
resource "google_project_iam_member" "workflows_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — act as the training SA when submitting Vertex AI CustomJobs
resource "google_service_account_iam_member" "workflows_act_as_training" {
  service_account_id = google_service_account.training.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workflows.email}"
}

# GCS service agent — publish events to Pub/Sub (required for Eventarc GCS triggers)
resource "google_project_iam_member" "gcs_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com"
}
