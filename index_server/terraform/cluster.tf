resource "google_container_cluster" "cluster" {
  name     = "cl-${random_uuid.cluster_id.result}"
  location = var.zone

  initial_node_count = 1

  node_config {
    preemptible  = true
    disk_size_gb = 10
  }

  network = var.network

  master_auth {
    username = ""
    password = ""

    client_certificate_config {
      issue_client_certificate = true
    }
  }
}

data "google_client_config" "provider" {}

provider "kubernetes" {
  load_config_file = false

  host  = "https://${google_container_cluster.cluster.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.cluster.master_auth[0].cluster_ca_certificate,
  )
}
