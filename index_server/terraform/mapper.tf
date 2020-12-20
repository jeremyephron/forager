variable "mapper_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-mapper"
}

variable "mapper_num_nodes" {
  type    = number
  default = 50
}

variable "mapper_node_type" {
  type    = string
  default = "n2-highcpu-16"
}

variable "mapper_num_replicas_per_node" {
  type    = number
  default = 1
}

variable "mapper_cpus" {
  type    = number
  default = 14
}

locals {
  mapper_external_port = 5000
  mapper_internal_port = 5000
  mapper_app_name      = "mapper"
  mapper_disk_size_gb  = 10
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

          env {
            name = "CPUS"
            value = var.mapper_cpus
          }

          resources {
            limits {
              cpu    = var.mapper_cpus
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
          "cloud.google.com/gke-nodepool" = google_container_cluster.cluster.node_pool.0.name
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
  value = "http://${kubernetes_service.mapper_svc.load_balancer_ingress.0.ip}:${local.mapper_external_port}"
}

output "num_mappers" {
  value = var.mapper_num_nodes * var.mapper_num_replicas_per_node
}
