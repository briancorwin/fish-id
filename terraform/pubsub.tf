resource "google_pubsub_topic" "analytics_events" {
  name       = "fish-id-analytics-events"
  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Main app's Cloud Run SA — publish only, scoped to this one topic.
resource "google_pubsub_topic_iam_member" "cloud_run_publisher" {
  topic  = google_pubsub_topic.analytics_events.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Pub/Sub's service agent is auto-provisioned when the API is enabled, with a
# well-known email of this form — no google-beta provider needed just to look it up.
locals {
  pubsub_service_agent = "service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Lets Pub/Sub mint OIDC tokens impersonating the consumer's own SA for authenticated push.
resource "google_service_account_iam_member" "pubsub_token_creator" {
  service_account_id = google_service_account.analytics_consumer.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${local.pubsub_service_agent}"

  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Push subscription — created only once the consumer's URL is known.
# See variables.tf (analytics_consumer_url) for the two-phase apply this requires.
resource "google_pubsub_subscription" "analytics_events_push" {
  count = var.analytics_consumer_url != "" ? 1 : 0

  name  = "fish-id-analytics-events-push"
  topic = google_pubsub_topic.analytics_events.name

  ack_deadline_seconds = 60

  push_config {
    push_endpoint = "${var.analytics_consumer_url}/push"
    oidc_token {
      service_account_email = google_service_account.analytics_consumer.email
    }
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  depends_on = [google_project_service.apis["pubsub.googleapis.com"]]
}

# Grants the consumer's own SA (used as the OIDC push identity) invoker rights
# on the Cloud Run service. The service itself is deployed imperatively via
# gcloud/CI, not Terraform-managed, but an IAM member resource can target it by
# name/location alone.
resource "google_cloud_run_v2_service_iam_member" "pubsub_invoker" {
  count = var.analytics_consumer_url != "" ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = "fish-id-analytics-consumer"
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.analytics_consumer.email}"
}
