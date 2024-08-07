/******************************************
	Google provider configuration
 *****************************************/

provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}

/******************************************
	Variables
 *****************************************/
variable "project" {
  description = "GCP project name"
  type        = string
}

variable "region" {
  description = "Default GCP region for resources"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "Default GCP zone for resources"
  type        = string
  default     = "europe-west1-b"
}
