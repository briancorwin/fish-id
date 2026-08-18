variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and Artifact Registry"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repository in owner/name format (e.g. briancorwin/fish-id)"
  type        = string
}

# Bootstrap note: leave this at its default empty string on the first `terraform
# apply`. It creates the Pub/Sub topic, BigQuery dataset/table, GCS bucket, and
# the analytics-consumer service account, but skips the push subscription and
# its Cloud Run invoker binding (both depend on a URL that doesn't exist yet).
# Deploy analytics-consumer once to obtain its URL, then re-apply with:
#   terraform apply -var="analytics_consumer_url=https://fish-id-analytics-consumer-xxxx-uc.a.run.app"
variable "analytics_consumer_url" {
  description = "URL of the deployed analytics-consumer Cloud Run service, used as the Pub/Sub push endpoint. Leave empty on first apply — see bootstrap note above."
  type        = string
  default     = ""
}
