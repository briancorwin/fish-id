locals {
  apis = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "firebase.googleapis.com",
    "firebasehosting.googleapis.com",
  ])
}

resource "google_project_service" "apis" {
  for_each = local.apis

  service            = each.key
  disable_on_destroy = false
}
