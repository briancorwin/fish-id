# Secret Manager shell for the GitHub PAT — operator adds the value manually
resource "google_secret_manager_secret" "github_deploy_pat" {
  secret_id = "github-deploy-pat"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

# Cloud Workflows definition for the continuous training pipeline
resource "google_workflows_workflow" "training_pipeline" {
  name            = "fish-id-training-pipeline"
  region          = var.region
  service_account = google_service_account.workflows.email
  source_contents = file("${path.module}/../workflows/fish-id-training-pipeline.yaml")

  depends_on = [google_project_service.apis["workflows.googleapis.com"]]
}

# Eventarc trigger — fires when a versioned manifest is finalized in the training bucket
resource "google_eventarc_trigger" "training_trigger" {
  name     = "fish-id-training-trigger"
  location = var.region

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.training.name
  }

  destination {
    workflow = google_workflows_workflow.training_pipeline.id
  }

  service_account = google_service_account.workflows.email

  depends_on = [google_project_service.apis["eventarc.googleapis.com"]]
}
