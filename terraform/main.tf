terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # bucket is passed at init time because backend blocks don't support variable interpolation.
  # One-time bootstrap (before first terraform init):
  #   gcloud storage buckets create gs://PROJECT_ID-fish-id-terraform-state \
  #     --location=REGION --uniform-bucket-level-access
  # Then:
  #   terraform init -backend-config="bucket=PROJECT_ID-fish-id-terraform-state"
  # After apply, import the bucket so Terraform manages it:
  #   terraform import google_storage_bucket.terraform_state PROJECT_ID-fish-id-terraform-state
  backend "gcs" {
    prefix = "fish-id"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
