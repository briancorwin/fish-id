resource "google_bigquery_dataset" "analytics" {
  dataset_id = "fish_id_analytics"
  location   = var.region
  depends_on = [google_project_service.apis["bigquery.googleapis.com"]]
}

resource "google_bigquery_table" "detection_events" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  table_id   = "detection_events"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  schema = jsonencode([
    { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "image_hash", type = "STRING", mode = "REQUIRED" },
    { name = "fish_count", type = "INTEGER", mode = "REQUIRED" },
    { name = "detections", type = "JSON", mode = "NULLABLE" },
    { name = "low_confidence", type = "BOOLEAN", mode = "REQUIRED" },
    { name = "image_stored", type = "BOOLEAN", mode = "REQUIRED" },
  ])

  depends_on = [google_project_service.apis["bigquery.googleapis.com"]]
}
