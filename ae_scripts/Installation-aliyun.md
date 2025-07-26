# Install HydraServe with Aliyun ACK

HydraServe is originally tested with [Aliyun ACK](https://www.alibabacloud.com/en/product/kubernetes?_p_lc=1). Follow the instructions below to prepare the cluster for HydraServe.

### 1. Create an Aliyun ACK Cluster

Create an ACK cluster by clicking **Create Kubernetes Cluster** at the [ACK Console](https://cs.console.aliyun.com/?#/k8s/cluster/list) page.

#### 1.1 Cluster Configurations

- Kubernetes Version: v1.30.7+
- Network Settings
  - Configure SNAT for VPC: Enable
  - Network Plug-in: Terway

#### 1.2 Node Pool Configurations

Create a node pool with zero initial nodes.
- Container Runtime: containerd
- Configure Managed Node Pool: Disable
- Volumes
  - System Disk: 400 GiB
- Instances
  - Expected Nodes: 0

#### 1.3 Component Configurations

You can disable all features.

### 2. Add ECS Instances

#### 2.1 Create ECS Instances

Create ECS instances by clicking **Create Instance** on the [ECS Console](https://ecs.console.aliyun.com/home#/) page.
- Image: Alibaba Cloud Linux
- Enable a public IP address for access
- Use the same VPC and security group as your ACK cluster
- All instances should be within the same region
- Allocate at least 400GB ESSD to store images

ECS types used in our latency measurement experiments (testbed (i)).

```
4 * ecs.gn7i-c32g1.8xlarge (1*A10)
4 * ecs.gn6e-c12g1.12xlarge (4*V100)
8 * ecs.c7a.16xlarge (16Gbps, As remote storage)
```

ECS types used in our end-to-end experiments (testbed (ii)).

```
2 * ecs.gn7i-c32g1.32xlarge (4*A10)
4 * ecs.gn6e-c12g1.12xlarge (4*V100)
12 * c7a.16xlarge (16Gbps, As remote storage)
```

#### 2.2 Add Instances into ACK Cluster

1. In the Cluster Management page, click Nodes -> Nodes Pools -> Add Existing Node for default nodepool.
2. Select the created instances.
3. Check **Store Container and Image Data on a Data Disk**.

#### 2.3 Configure Kubernetes accessibility

1. In the Cluster Management page, click Cluster Information -> Connection Information -> Obtain Long-term Kubeconfig, and copy the content in **Internal Access**.
2. Log in the master node (select an arbitriry node that does not perform GPU inference), and run the following commands.
```
mkdir -p ~/.kube
vim ~/.kube/config
[Paste the just copied content into it]
```

### 3. Enable GPU Sharing

1. In the Cluster Managemenet page, click Applications -> Cloud-native AI Suite -> Deploy.
2. Check **Scheduling Policy Extension**.
3. Click **Advanced**, and configure the `policy` field of `cgpu` to 1.
4. Deploy the suite.

### 4. Configure Node Label
   
Log in to the master node, and label all GPU servers with the following command.
```
kubectl label [node_name] ack.node.gpu.schedule=cgpu --overwrite
```

If you are using ECS instances not listed in 2.2, please configure the specifications of instance types in `scripts/kubernetes/vllm/src/ECSInstance.py`.

### 5. Initialize Master Environment

Log in to your master node and run the following commands.
```
[Clone this repo]
cd hydraserve/scripts/kubernetes
sh install_python.sh    # The kubernetes package version must be consistent with the version of your local kubernetes.
sh tool-node-shell/setup.sh
```