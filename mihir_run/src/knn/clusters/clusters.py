import asyncio
import uuid

from typing import Optional, Tuple

import aiohttp
from gcloud.aio.auth import Token
from kubernetes_asyncio import client
from kubernetes_asyncio.client.api_client import ApiClient


DISK_SIZE_GB = 10
DISK_TYPE = "pd-ssd"
GKE_VERSION = "1.16.13-gke.401"
USE_PREEMPTIBLES = True
POLL_INTERVAL = 2
INTERNAL_PORT = 5000
EXTERNAL_START_PORT = 5000
KUBE_NAMESPACE = "default"


class GKECluster:  # TODO(mihirg): Rename to GKENodePool
    def __init__(
        self,
        project_id: str,
        zone: str,
        cluster_name: str,
        machine_type: str,
        num_nodes: int,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.project_id = project_id
        self.zone = zone
        self.cluster_name = cluster_name
        self.machine_type = machine_type
        self.num_nodes = num_nodes

        self._unassigned_port = EXTERNAL_START_PORT

        self.started = False
        self.cluster_id = f"np-{uuid.uuid4()}"  # TODO(mihirg): Change to pool_id

        # Will be initialized later
        self.session: Optional[aiohttp.ClientSession] = session
        self.auth_client: Optional[Token] = None
        self.cluster_endpoint: Optional[str] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.stop()

    def get_unassigned_port(self):
        self._unassigned_port += 1
        return self._unassigned_port - 1

    # CLUSTER MANAGEMENT

    async def update_cluster_endpoint(self):
        endpoint = (
            "https://container.googleapis.com/v1/"
            f"projects/{self.project_id}/"
            f"zones/{self.zone}/"
            f"clusters/{self.cluster_name}"
        )
        headers = {"Authorization": f"Bearer {await self.auth_client.get()}"}

        while not self.cluster_endpoint:
            await asyncio.sleep(POLL_INTERVAL)
            async with self.session.get(endpoint, headers=headers) as response:
                if response.status != 200:
                    print(response, await response.text())
                    continue

                response_json = await response.json()
                if response_json.get("status") == "RUNNING":
                    self.cluster_endpoint = response_json.get("endpoint")

    async def start(self, num_retries=15):
        self.session = self.session or aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False)
        )
        self.auth_client = Token(session=self.session)

        # Create node pool
        endpoint = (
            "https://container.googleapis.com/v1/"
            f"projects/{self.project_id}/"
            f"zones/{self.zone}/"
            f"clusters/{self.cluster_name}/"
            "nodePools"
        )
        request = {
            "nodePool": {
                "name": self.cluster_id,
                "config": {
                    "machineType": self.machine_type,
                    "diskSizeGb": DISK_SIZE_GB,
                    "oauthScopes": [
                        "https://www.googleapis.com/auth/devstorage.read_only",
                        "https://www.googleapis.com/auth/logging.write",
                        "https://www.googleapis.com/auth/monitoring",
                        "https://www.googleapis.com/auth/servicecontrol",
                        "https://www.googleapis.com/auth/service.management.readonly",  # noqa
                        "https://www.googleapis.com/auth/trace.append",
                    ],
                    "metadata": {"disable-legacy-endpoints": "true"},
                    "preemptible": USE_PREEMPTIBLES,
                    "diskType": DISK_TYPE,
                    "shieldedInstanceConfig": {"enableIntegrityMonitoring": True},
                },
                "initialNodeCount": self.num_nodes,
                "autoscaling": {},
                "management": {"autoUpgrade": True, "autoRepair": True},
                "version": GKE_VERSION,
                "upgradeSettings": {"maxSurge": 1},
            }
        }
        headers = {"Authorization": f"Bearer {await self.auth_client.get()}"}
        for i in range(num_retries):
            async with self.session.post(
                endpoint, json=request, headers=headers
            ) as response:
                if response.status == 400 and (i + 1) < num_retries:  # retry
                    await asyncio.sleep(POLL_INTERVAL)
                elif response.status != 200:
                    print(response, await response.text())
                    raise RuntimeError("Error when attempting to create node pool")
                else:
                    break

        # Wait for node pool to be fully created (and master to resize if necessary)
        await self.update_cluster_endpoint()

        print(f"Started Kubernetes node pool {self.cluster_id}")
        self.started = True

    async def scale(self, num_nodes: int):
        assert self.session
        assert self.auth_client
        assert self.started  # node pool has been started

        endpoint = (
            "https://container.googleapis.com/v1/"
            f"projects/{self.project_id}/"
            f"zones/{self.zone}/"
            f"clusters/{self.cluster_name}"
            f"nodePools/{self.cluster_id}"
        )
        request = {
            "nodeCount": num_nodes,
        }

        headers = {"Authorization": f"Bearer {await self.auth_client.get()}"}
        async with self.session.post(
            endpoint, json=request, headers=headers
        ) as response:
            assert response.status == 200, response.status

        self.num_nodes = num_nodes

    async def stop(self):
        assert self.session
        assert self.auth_client

        if self.started:
            endpoint = (
                "https://container.googleapis.com/v1/"
                f"projects/{self.project_id}/"
                f"zones/{self.zone}/"
                f"clusters/{self.cluster_name}/"
                f"nodePools/{self.cluster_id}"
            )

            headers = {"Authorization": f"Bearer {await self.auth_client.get()}"}
            async with self.session.delete(endpoint, headers=headers) as response:
                assert response.status == 200, response.status

            # Wait for node pool to be fully deleted
            await self.update_cluster_endpoint()

        await self.session.close()

    # DEPLOYMENT MANAGEMENT

    async def create_deployment(
        self,
        container: str,
        num_replicas: int,
        cpus: float = 1.0,
        memory: float = 1.0,
    ) -> Tuple[str, str]:
        assert self.auth_client
        assert self.cluster_endpoint

        cfg = client.Configuration(
            host=f"https://{self.cluster_endpoint}:443",
            api_key={"authorization": f"Bearer {await self.auth_client.get()}"},
        )
        cfg.verify_ssl = False

        async with ApiClient(configuration=cfg) as kube_api:
            apps_api = client.AppsV1Api(kube_api)
            core_api = client.CoreV1Api(kube_api)

            # Create deployment
            deployment_id = f"dep-{uuid.uuid4()}"
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=deployment_id),
                spec=client.V1DeploymentSpec(
                    replicas=num_replicas,
                    selector={"matchLabels": {"dep": deployment_id}},
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"dep": deployment_id}),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=deployment_id,
                                    env=[
                                        client.V1EnvVar(
                                            name="PORT", value=str(INTERNAL_PORT)
                                        )
                                    ],
                                    image=container,
                                    resources=client.V1ResourceRequirements(
                                        requests={
                                            "cpu": str(cpus),
                                            "memory": f"{int(memory * 1024)}M",
                                        }
                                    ),
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=INTERNAL_PORT
                                        )
                                    ],
                                )
                            ],
                            node_selector={
                                "cloud.google.com/gke-nodepool": self.cluster_id
                            },
                        ),
                    ),
                ),
            )
            await apps_api.create_namespaced_deployment(
                namespace=KUBE_NAMESPACE, body=deployment
            )

            # Create service
            service_id = f"{deployment_id}-svc"
            service_port = self.get_unassigned_port()
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=service_id),
                spec=client.V1ServiceSpec(
                    selector={"dep": deployment_id},
                    ports=[
                        client.V1ServicePort(
                            protocol="TCP",
                            port=service_port,
                            target_port=INTERNAL_PORT,
                        )
                    ],
                    type="LoadBalancer",
                ),
            )
            await core_api.create_namespaced_service(
                namespace=KUBE_NAMESPACE, body=service
            )

            # Poll for external URL
            service_ip = None
            while not service_ip:
                await asyncio.sleep(POLL_INTERVAL)
                ingress = (
                    await core_api.read_namespaced_service(
                        name=service_id, namespace=KUBE_NAMESPACE
                    )
                ).status.load_balancer.ingress
                if ingress:
                    service_ip = ingress[0].ip

        service_url = f"http://{service_ip}:{service_port}"
        print(f"Started deployment {deployment_id} at {service_url}")

        return deployment_id, service_url

    async def delete_deployment(self, deployment_id: str):
        assert self.auth_client
        assert self.cluster_endpoint

        cfg = client.Configuration(
            host=f"https://{self.cluster_endpoint}:443",
            api_key={"authorization": f"Bearer {await self.auth_client.get()}"},
        )
        cfg.verify_ssl = False

        async with ApiClient(configuration=cfg) as kube_api:
            apps_api = client.AppsV1Api(kube_api)
            core_api = client.CoreV1Api(kube_api)

            # Delete service
            service_id = f"{deployment_id}-svc"
            await core_api.delete_namespaced_service(
                name=service_id, namespace=KUBE_NAMESPACE
            )

            # Delete deployment
            await apps_api.delete_namespaced_deployment(
                name=deployment_id, namespace=KUBE_NAMESPACE
            )
