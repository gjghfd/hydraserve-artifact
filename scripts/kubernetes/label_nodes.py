import os
import sys
import time
import asyncio
from kubernetes import client, config

nas_path = "/mnt"
share = True if int(os.getenv("SHARE", "1")) == 1 else False

if __name__ == '__main__':
    config.load_kube_config()
    core_api = client.CoreV1Api()

    print(f"Share = {share}")
    label = "cgpu" if share else "default"

    node_list = core_api.list_node()
    for node in node_list.items:
        node_labels = node.metadata.labels
        if "aliyun.accelerator/nvidia_name" in node_labels:
            node_name = node_labels["kubernetes.io/hostname"]
            print(f"Label node {node_name}")
            os.system(f"kubectl label nodes {node_name} ack.node.gpu.schedule={label} --overwrite")