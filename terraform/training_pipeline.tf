# Secret Manager shell for the GitHub PAT — operator adds the value manually
resource "google_secret_manager_secret" "github_deploy_pat" {
  secret_id = "github-deploy-pat"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

# Zip the Cloud Functions trigger source so Cloud Build can deploy it
data "archive_file" "pipeline_trigger" {
  type        = "zip"
  source_dir  = "${path.module}/../pipeline/trigger"
  output_path = "/tmp/fish-id-pipeline-trigger.zip"
}

resource "google_storage_bucket_object" "pipeline_trigger_source" {
  name   = "pipeline-trigger-${data.archive_file.pipeline_trigger.output_md5}.zip"
  bucket = google_storage_bucket.training.name
  source = data.archive_file.pipeline_trigger.output_path
}

# Cloud Functions v2: receives GCS finalise events and submits a Vertex AI Pipeline run
resource "google_cloudfunctions2_function" "pipeline_trigger" {
  name     = "fish-id-pipeline-trigger"
  location = var.region

  build_config {
    runtime     = "python311"
    entry_point = "trigger_pipeline"
    source {
      storage_source {
        bucket = google_storage_bucket.training.name
        object = google_storage_bucket_object.pipeline_trigger_source.name
      }
    }
  }

  service_config {
    max_instance_count             = 1
    min_instance_count             = 0
    available_memory               = "256M"
    timeout_seconds                = 60
    service_account_email          = google_service_account.workflows.email
    all_traffic_on_latest_revision = true
    environment_variables = {
      GCP_PROJECT_ID        = var.project_id
      GCP_REGION            = var.region
      TRAINING_BUCKET       = google_storage_bucket.training.name
      MODEL_BUCKET          = google_storage_bucket.models.name
      PIPELINE_SA           = google_service_account.workflows.email
      PIPELINE_TEMPLATE_URI = "gs://${google_storage_bucket.models.name}/pipeline/fish-id-training-pipeline.json"
      VERTEX_EXPERIMENT     = "fish-id-eval"
    }
  }

  # Eventarc trigger: fires on every GCS object finalisation in the training bucket.
  # The function itself filters for versioned manifests (versions/*/manifest.json).
  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.storage.object.v1.finalized"
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.workflows.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.training.name
    }
  }

  depends_on = [
    google_project_service.apis["cloudfunctions.googleapis.com"],
    google_storage_bucket_object.pipeline_trigger_source,
  ]
}

# Allow Eventarc (via the workflows SA) to invoke the Cloud Run service backing the function
resource "google_cloud_run_v2_service_iam_member" "trigger_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloudfunctions2_function.pipeline_trigger.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflows.email}"
}
