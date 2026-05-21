output "workload_identity_provider" {
  description = "Value for the GCP_WORKLOAD_IDENTITY_PROVIDER GitHub Actions secret"
  value       = "projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/providers/${google_iam_workload_identity_pool_provider.github.workload_identity_pool_provider_id}"
}

output "cicd_service_account_email" {
  description = "Value for the GCP_SERVICE_ACCOUNT GitHub Actions secret"
  value       = google_service_account.cicd.email
}

output "model_bucket_name" {
  description = "Upload best.onnx here, then set ONNX_MODEL_GCS_URI=gs://BUCKET/best.onnx in GitHub secrets"
  value       = google_storage_bucket.models.name
}

output "artifact_registry_repo" {
  description = "Docker image prefix used in deploy.yml (REGION-docker.pkg.dev/PROJECT/fish-id)"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fish_id.repository_id}"
}
