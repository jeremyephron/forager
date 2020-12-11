variable "mapper_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-mapper"
}

variable "mapper_num_nodes" {
  type    = number
  default = 25
}

variable "mapper_node_type" {
  type    = string
  default = "n2-highcpu-16"
}

variable "mapper_num_replicas_per_node" {
  type    = number
  default = 12
}

variable "mapper_ram_gb" {
  type    = number
  default = 1
}

variable "mapper_cpus" {
  type    = number
  default = 1
}

locals {
  mapper_external_port = 5000
  mapper_internal_port = 5000
  mapper_app_name      = "mapper"
  mapper_disk_size_gb  = 10
}

resource "google_container_node_pool" "mapper_np" {
  name       = "mapper-np"
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
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

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubernetes_deployment" "mapper_dep" {
  metadata {
    name = "mapper-dep"
    labels = {
      app = local.mapper_app_name
    }
  }

  spec {
    replicas = var.mapper_num_nodes * var.mapper_num_replicas_per_node

    selector {
      match_labels = {
        app = local.mapper_app_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.mapper_app_name
        }
      }

      spec {
        container {
          image = var.mapper_image_name
          name  = local.mapper_app_name

          env {
            name  = "PORT"
            value = local.mapper_internal_port
          }

          resources {
            limits {
              cpu    = var.mapper_cpus
              memory = "${floor(var.mapper_ram_gb * 1024)}Mi"
            }
          }

          port {
            container_port = local.mapper_internal_port
          }

          volume_mount {
            mount_path = local.nfs_mount_dir
            name       = local.nfs_volume_name
          }
        }

        volume {
          name = local.nfs_volume_name

          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name
          }
        }

        node_selector = {
          "cloud.google.com/gke-nodepool" = google_container_node_pool.mapper_np.name
        }
      }
    }
  }
}

resource "kubernetes_service" "mapper_svc" {
  metadata {
    name = "${kubernetes_deployment.mapper_dep.metadata.0.name}-svc"
  }
  spec {
    selector = kubernetes_deployment.mapper_dep.metadata.0.labels
    port {
      port        = local.mapper_external_port
      target_port = local.mapper_internal_port
    }

    type = "LoadBalancer"
  }
}

output "mapper_url" {
  value = "https://${kubernetes_service.mapper_svc.load_balancer_ingress.0.ip}:${local.mapper_external_port}"
}

output "num_mappers" {
  value = var.mapper_num_nodes * var.mapper_num_replicas_per_node
}
