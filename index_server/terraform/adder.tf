variable "adder_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-adder"
}

variable "adder_num_nodes" {
  type    = number
  default = 50
}

variable "adder_node_type" {
  type    = string
  default = "n2-highcpu-4"
}

locals {
  adder_external_port = 5000
  adder_internal_port = 5000
  adder_app_name      = "adder"
  adder_disk_size_gb  = 10
}

resource "kubernetes_deployment" "adder_dep" {
  metadata {
    name = "adder-dep"
    labels = {
      app = local.adder_app_name
    }
  }

  spec {
    replicas = var.adder_num_nodes

    selector {
      match_labels = {
        app = local.adder_app_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.adder_app_name
        }
      }

      spec {
        container {
          image = var.adder_image_name
          name  = local.adder_app_name

          env {
            name  = "PORT"
            value = local.adder_internal_port
          }

          port {
            container_port = local.adder_internal_port
          }

          volume_mount {
            mount_path = local.nfs_mount_dir
            name       = local.nfs_volume_name
          }
        }

        affinity {
          pod_anti_affinity {
            required_during_scheduling_ignored_during_execution {
              label_selector {
                match_expressions {
                  key      = "app"
                  operator = "In"
                  values   = [local.adder_app_name]
                }
              }

              topology_key = "kubernetes.io/hostname"
            }
          }
        }

        volume {
          name = local.nfs_volume_name

          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name
          }
        }

        node_selector = {
          "cloud.google.com/gke-nodepool" = google_container_cluster.cluster.node_pool.1.name
        }
      }
    }
  }
}

resource "kubernetes_service" "adder_svc" {
  metadata {
    name = "${kubernetes_deployment.adder_dep.metadata.0.name}-svc"
  }
  spec {
    selector = kubernetes_deployment.adder_dep.metadata.0.labels
    port {
      port        = local.adder_external_port
      target_port = local.adder_internal_port
    }

    type = "LoadBalancer"
  }
}

output "adder_url" {
  value = "https://${kubernetes_service.adder_svc.load_balancer_ingress.0.ip}:${local.adder_external_port}"
}

output "num_adders" {
  value = var.adder_num_nodes
}
