variable "bgsplit_mapper_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-bgsplit-mapper:latest"
}

variable "bgsplit_mapper_node_pool_name" {
  type    = string
  default = "bgsplit-mapper-np"
}

variable "bgsplit_mapper_num_nodes" {
  type    = number
  default = 8
}

variable "bgsplit_mapper_node_type" {
  type    = string
  default = "n1-standard-8"
}

variable "bgsplit_mapper_nproc" {
  type    = number
  default = 1
}

variable "bgsplit_mapper_accelerator_type" {
  type    = string
  default = "nvidia-tesla-t4"
}

variable "bgsplit_mapper_accelerator_count" {
  type    = number
  default = 1
}

variable "bgsplit_mapper_gpus" {
  type    = number
  default = 4
}

locals {
  bgsplit_mapper_external_port = 5000
  bgsplit_mapper_internal_port = 5000
  bgsplit_mapper_app_name      = "bgsplit-mapper"
  bgsplit_mapper_disk_size_gb  = 10
}

resource "google_container_node_pool" "bgsplit_mapper_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.bgsplit_mapper_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.bgsplit_mapper_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.bgsplit_mapper_node_type
    disk_size_gb = local.bgsplit_mapper_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes

    guest_accelerator {
      type  = var.bgsplit_mapper_accelerator_type
      count = var.bgsplit_mapper_accelerator_count
    }
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubectl_manifest" "bgsplit_mapper_dep" {
  count = var.bgsplit_mapper_num_nodes

  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bgsplit-mapper-dep-${count.index}
  labels:
    app: ${local.bgsplit_mapper_app_name}-${count.index}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${local.bgsplit_mapper_app_name}-${count.index}
  template:
    metadata:
      labels:
        app: ${local.bgsplit_mapper_app_name}-${count.index}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ${local.bgsplit_mapper_app_name}
            topologyKey: kubernetes.io/hostname
      containers:
      - image: ${var.bgsplit_mapper_image_name}
        name: ${local.bgsplit_mapper_app_name}
        resources:
          limits:
            nvidia.com/gpu: ${var.bgsplit_mapper_gpus}
        env:
        - name: PORT
          value: "${local.bgsplit_mapper_internal_port}"
        ports:
        - containerPort: ${local.bgsplit_mapper_internal_port}
        volumeMounts:
        - mountPath: ${local.nfs_mount_dir}
          name: ${local.nfs_volume_name}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${var.bgsplit_mapper_node_pool_name}
      tolerations:
      - effect: NoSchedule
        key: nvidia.com/gpu
        operator: Exists
      volumes:
      - name: ${local.nfs_volume_name}
        persistentVolumeClaim:
          claimName: ${kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name}
YAML

  depends_on = [google_container_cluster.cluster, google_container_node_pool.bgsplit_mapper_np]
}

resource "kubernetes_service" "bgsplit_mapper_svc" {
  count = var.bgsplit_mapper_num_nodes

  metadata {
    name = "bgsplit-mapper-dep-${count.index}-svc"
  }
  spec {
    selector = {
      app = "${local.bgsplit_mapper_app_name}-${count.index}"
    }
    port {
      port        = local.bgsplit_mapper_external_port
      target_port = local.bgsplit_mapper_internal_port
    }

    type = "LoadBalancer"
  }
}

output "bgsplit_mapper_url" {
  value = [
    for svc in kubernetes_service.bgsplit_mapper_svc:
    "http://${svc.status.0.load_balancer.0.ingress.0.ip}:${local.bgsplit_mapper_external_port}"
  ]
}

output "bgsplit_num_mappers" {
  value = var.bgsplit_mapper_num_nodes
}

output "bgsplit_mapper_nproc" {
  value = var.bgsplit_mapper_nproc
}
