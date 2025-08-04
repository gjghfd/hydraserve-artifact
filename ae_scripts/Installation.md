# 1. Installation

### 1.1 Prepare Images

We provide built images listed as below.

```
Modified vLLM: registry.us-east-1.aliyuncs.com/kubernetes-fc/modelscope-vllm:v1

Model Downloader: registry.cn-shanghai.aliyuncs.com/kubernetes-fc/vllm-download:v1

Storage Server:
- registry.us-east-1.aliyuncs.com/kubernetes-fc/vllm-storage-server:v1
- chlou/vllm-storage-server-remote:v1
```

#### Build Images from Source

1. Build modified vLLM (replace [IMAGE_NAME] with your desired Docker image name)

```
DOCKER_BUILDKIT=1 docker build . --platform=linux/amd64 --tag [IMAGE_NAME] -f scripts/image/Dockerfile
```

2. Build model downloader

```
cd scripts/kubernetes/vllm/download
docker build . --platform=linux/amd64 --tag [IMAGE_NAME]
```

3. Build storage server

```
cd scripts/kubernetes/vllm/storage_server
docker build . --platform=linux/amd64 --tag [IMAGE_NAME_LOCAL_SERVER]
docker build . --platform=linux/amd64 --tag [IMAGE_NAME_REMOTE_SERVER] -f Dockerfile_remote
```

If you build images from source, please configure the image names in `scripts/kubernetes/vllm/src/ImageInfo.py`.

### 1.2 Prepare Environments

If you are using Aliyun ACK Cluster, refer to the environment preparation guide in [Installation-aliyun](Installation-aliyun.md).

1. Environment Requirements

HydraServe needs at least one GPU server to perform inference, and at least one server to act as remote storage.
- Kubernetes v1.30.7+
- Python 3.8+

Log in to the master of your Kubernetes cluster and run the following commands.
```
# The kubernetes package version must be consistent with the version of your local kubernetes.
pip install kubernetes==30.1.0 modelscope==1.15.0 requests openai fastapi aiohttp uvicorn[standard]
sh scripts/kubernetes/tool-node-shell/setup.sh
```

2. Enable GPU Sharing

Install the [Aliyun GPUShare Plugin](https://github.com/AliyunContainerService/gpushare-scheduler-extender) to enable GPU sharing.

3. Configure Node Label
   
Label all GPU servers.
```
kubectl label [node_name] gpushare=true --overwrite
```

Label all servers with specifications.
```
kubectl label [node_name] node.kubernetes.io/instance-type=[Instance Type] --overwrite
```
Configure the specifications of instance types in scripts/kubernetes/vllm/src/ECSInstance.py.

### 1.3 Fetch Images and Download Models

Run the following commands to fetch the required Docker images and download models. You can obtain your modelscope token from https://www.modelscope.cn/my/myaccesstoken
```
cd scripts/kubernetes/vllm
python src/init_images.py           # fetch images
export MODEL_DIR=[PATH_TO_MODEL_DIR]
export MODELSCOPE_TOKEN=[MODELSCOPE_ACCESS_TOKEN]
# If you are using a NAS shared by all servers, configure the USE_NAS environment variable to 1
export USE_NAS=1
python src/init_models.py           # download models from modelscope
python src/init_shm.py              # init shared memory of gpu nodes
```

To download all models required in our cold-start latency experiment (Section 8.2 in the paper), run
```
MODEL_SET=2 python src/init_models.py
```

If you have downloaded models on the current server, and want to broadcast it to all other servers, run
```
export MODEL_DIR=[PATH_TO_MODEL_DIR]
python src/broadcast_model.py model-cache
```

# 2. Run HydraServe

## 2.1 Start HydraServe Endpoint

```
cd scripts/kubernetes/vllm
# This step enables local memory cache and is optional.
export USE_CACHE=1
export MODEL_DIR=[PATH_TO_MODEL_DIR]
export LOG_PATH=[PATH_TO_LOG_DIR]           
# Start storage server
python src/start_storage_server.py  
# Start HydraServe endpoint
python src/main.py                  
```

HydraServe endpoint runs on localhost:9090.

To stop storage server, run the following command:
```
python src/close_storage_server.py
```

### 2.2 Testing

1. Send a chat request to HydraServe
```
curl -X POST -H "Content-Type: application/json" -d "{\"id\": \"0\", \"model\": \"modelscope/Llama-2-7b-chat-ms/0\", \"prompt\": \"hello\"}"  http://0.0.0.0:9090
```

2. Generate traces and run an end-to-end experiment
```
[Follow the instructions in ./trace to generate the workload.]

# Use request generator
cd scripts/kubernetes/vllm
python src/request_generator.py [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
```

3. Analyze result
```
cd scripts/kubernetes/vllm
python src/result_analyzer.py [PATH_TO_REQUEST_GENERATOR_OUTPUT_LOG]

# Analyze Cost
python src/result_analyzer.py [PATH_TO_REQUEST_GENERATOR_OUTPUT_LOG] [PATH_TO_HYDRASERVE_OUTPUT_LOG] [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
```
