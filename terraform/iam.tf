data "google_project" "project" {}

# Cloud Run service account — publishes analytics events to Pub/Sub (see pubsub.tf)
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

# CI/CD SA — attach fish-id-cloud-run-sa when deploying (deploy-api.yml)
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

# CI/CD SA — submit Vertex AI training pipeline runs (train-pipeline.yml) and read the Model
# Registry to resolve the production alias at deploy time (aiplatform.user covers both)
resource "google_project_iam_member" "cicd_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cicd.email}"
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

# Workflows SA — read training data (container components run under the pipeline SA, not training SA)
resource "google_storage_bucket_iam_member" "workflows_training_bucket_reader" {
  bucket = google_storage_bucket.training.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.workflows.email}"
}

# Workflows SA — write logs
resource "google_project_iam_member" "workflows_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.workflows.email}"
}

# CI/CD SA — attach fish-id-workflows-sa as the pipeline's runtime service account when
# submitting a run (train-pipeline.yml, aiplatform.PipelineJob.submit(service_account=...))
resource "google_service_account_iam_member" "cicd_workflows_sa_user" {
  service_account_id = google_service_account.workflows.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd.email}"
}

# Secret holding the GitHub PAT used to trigger workflow_dispatch on deploy-api.yml
# The secret version (the PAT value) must be added manually after `terraform apply`:
#   echo -n "YOUR_PAT" | gcloud secrets versions add fish-id-github-deploy-token --data-file=-
resource "google_secret_manager_secret" "github_deploy_token" {
  secret_id = "fish-id-github-deploy-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

# Workflows SA — read the GitHub deploy token at pipeline runtime
resource "google_secret_manager_secret_iam_member" "workflows_deploy_token" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.github_deploy_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workflows.email}"
}

# Analytics consumer service account — receives Pub/Sub push, writes BigQuery + GCS
resource "google_service_account" "analytics_consumer" {
  account_id   = "fish-id-analytics-consumer-sa"
  display_name = "fish-id Analytics Consumer SA"
  depends_on   = [google_project_service.apis["iam.googleapis.com"]]
}

# Sufficient for insert_rows_json (tabledata.insertAll) — no bigquery.jobUser needed
resource "google_bigquery_dataset_iam_member" "consumer_bq_writer" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.analytics_consumer.email}"
}

resource "google_storage_bucket_iam_member" "consumer_images_writer" {
  bucket = google_storage_bucket.review_images.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.analytics_consumer.email}"
}

# CI/CD SA — attach fish-id-analytics-consumer-sa when deploying (deploy-analytics-consumer.yml)
resource "google_service_account_iam_member" "cicd_sa_user_analytics_consumer" {
  service_account_id = google_service_account.analytics_consumer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cicd.email}"
}
