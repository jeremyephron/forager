variable "trainer_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-trainer"
}

variable "trainer_num_nodes" {
  type    = number
  default = 4
}

variable "trainer_node_type" {
  type    = string
  default = "n1-highmem-8"
}

variable "trainer_accelerator_type" {
  type    = string
  default = "nvidia-tesla-k80"
}

variable "trainer_accelerator_count" {
  type    = number
  default = 1
}

variable "trainer_gpus" {
  type    = number
  default = 1
}

locals {
  trainer_external_port = 5000
  trainer_internal_port = 5000
  trainer_app_name      = "trainer"
  trainer_disk_size_gb  = 20
}

resource "kubernetes_deployment" "trainer_dep" {
  count = var.trainer_num_nodes

  metadata {
    name = "trainer-dep-${count.index}"
    labels = {
      app = "${local.trainer_app_name}-${count.index}"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "${local.trainer_app_name}-${count.index}"
      }
    }

    template {
      metadata {
        labels = {
          app = "${local.trainer_app_name}-${count.index}"
        }
      }

      spec {
        container {
          image = var.trainer_image_name
          name  = local.trainer_app_name

          env {
            name  = "PORT"
            value = local.trainer_internal_port
          }

          port {
            container_port = local.trainer_internal_port
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
                  values   = [local.trainer_app_name]
                }
              }

              topology_key = "kubernetes.io/hostname"
            }
          }
        }

        toleration {
          effect   = "NoSchedule"
          key      = "nvidia.com/gpu"
          operator = "Exists"
        }

        volume {
          name = local.nfs_volume_name

          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name
          }
        }

        node_selector = {
          "cloud.google.com/gke-nodepool" = google_container_cluster.cluster.node_pool.2.name
        }
      }
    }
  }
}

resource "kubernetes_service" "trainer_svc" {
  for_each = { for i, dep in kubernetes_deployment.trainer_dep : i => dep }

  metadata {
    name = "${each.value.metadata.0.name}-svc"
  }
  spec {
    selector = each.value.metadata.0.labels
    port {
      port        = local.trainer_external_port
      target_port = local.trainer_internal_port
    }

    type = "LoadBalancer"
  }
}

output "trainer_urls" {
  value = [
    for svc in kubernetes_service.trainer_svc:
    "http://${svc.load_balancer_ingress.0.ip}:${local.trainer_external_port}"
  ]
}
