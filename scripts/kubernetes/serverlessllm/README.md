# 1. Install ServerlessLLM with Aliyun ACK

This guide decribes how to install ServerlessLLM on Aliyun ACK cluster.
Please make sure you have installed HydraServe on an Aliyun ACK cluster by following [HydraServe Setup Guide](../../../README.md).

### 1.1 Build Docker Images

We provide built images listed as below.
```
registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve:v1
registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve-worker:v1
```

You can use `docker-build.sh` to build images on your own.
### 1.2 Prepare Environment

1. Create anaconda environment
```
sh setup.sh 0
```

2. Label all GPU servers
```
kubectl label [node_name] ack.node.gpu.schedule=default --overwrite
```

### 1.3 Fetch Images and Download Models

Run the following commands to fetch the required Docker images and download models.
```
python src/init_images.py           # init images
EXPR_1_1=1 python src/init_servers.py
# In another terminal
kubectl node-shell $(cat worker_name.txt) -- sh -c "ln -s /mnt/model-cache /mnt/sllm"   # link to models downloaded by HydraServe
conda activate sllm
export SERVER_POD_IP=$(cat head_ip.txt)
export LLM_SERVER_URL=http://$SERVER_POD_IP:8343/
python src/init_models.py           # create serverlessllm-specific model weights
# If you are not using NAS, run the following command to broadcast model weights to other servers
SRC=GPU python ../vllm/src/broadcast_models.py vllm
```

# 2. Run ServerlessLLM

### 2.1 Start ServerlessLLM Endpoint
```
python src/init_servers.py
```

### 2.2 Testing

1. Send a chat request to ServerlessLLM
```
conda activate sllm
export SERVER_POD_IP=$(cat head_ip.txt)
export LLM_SERVER_URL=http://$SERVER_POD_IP:8343/
sllm-cli deploy --config /root/model_configs/modelscope/Llama-2-7b-chat-ms/0/config.json 
sllm-cli deploy --config /root/model_configs/modelscope/Llama-2-13b-chat-ms/0/config.json 
sllm-cli deploy --config /root/model_configs/LLM-Research/Meta-Llama-3-8B-Instruct/0/config.json
curl http://$SERVER_POD_IP:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "modelscope/Llama-2-7b-chat-ms/0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ],
        "max_tokens": 1024
    }'
```

2. Run an end-to-end experiment
```
# 1. generate model file
python ../vllm/src/request_generator.py [PATH_TO_WORKLOAD] [REQ_PER_SECOND] models.txt

# 2. deploy models
conda activate sllm
export SERVER_POD_IP=$(cat head_ip.txt)
export LLM_SERVER_URL=http://$SERVER_POD_IP:8343/
# Deploy all models
python deploy_models.py models.txt  #It takes up to 5 minutes

# 3. run
export SERVERLESS_LLM=1
python ../vllm/src/request_generator.py [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
```
