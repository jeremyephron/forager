resource "google_container_cluster" "cluster" {
  name     = "cl-${random_uuid.cluster_id.result}"
  location = var.zone
  network  = var.network

  node_pool {
    name       = "mapper-np"
    node_count = var.mapper_num_nodes

    node_config {
      preemptible  = true
      machine_type = var.mapper_node_type
      disk_size_gb = local.trainer_disk_size_gb

      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",  # noqa
        "https://www.googleapis.com/auth/trace.append",
      ]
    }
  }

  node_pool {
    name       = "adder-np"
    node_count = var.adder_num_nodes

    node_config {
      preemptible  = true
      machine_type = var.adder_node_type
      disk_size_gb = local.adder_disk_size_gb

      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",  # noqa
        "https://www.googleapis.com/auth/trace.append",
      ]
    }
  }

  node_pool {
    name       = "trainer-np"
    node_count = var.trainer_num_nodes

    node_config {
      preemptible  = true
      machine_type = var.trainer_node_type
      disk_size_gb = local.trainer_disk_size_gb

      guest_accelerator {
        type  = var.trainer_accelerator_type
        count = var.trainer_accelerator_count
      }

      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",  # noqa
        "https://www.googleapis.com/auth/trace.append",
      ]
    }
  }

  master_auth {
    username = ""
    password = ""

    client_certificate_config {
      issue_client_certificate = true
    }
  }
}

data "google_client_config" "provider" {}

provider "kubectl" {
  load_config_file = false

  host  = "https://${google_container_cluster.cluster.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.cluster.master_auth[0].cluster_ca_certificate,
  )
}

# resource "kubectl_manifest" "gpu_installer" {
#   yaml_body = file("${path.module}/gpu_installer.yaml")
# }

provider "kubernetes" {
  load_config_file = false

  host  = "https://${google_container_cluster.cluster.endpoint}"
  token = data.google_client_config.provider.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.cluster.master_auth[0].cluster_ca_certificate,
  )
}
