expr=$1
exec_type=$2
model_set=$3
backend=$4
cv=$5
req_rate=$6

cur_dir=$(dirname "$0")
cd $cur_dir/../scripts/kubernetes/vllm

if [ "$expr" == "0" ]; then
    if [ "$backend" == "a10" ]; then
        if [ "$model_set" == "1" ]; then
            echo "No need to run model_set 1 on A10 GPUs. Please move on to next setting."
            exit 1
        fi
        used_machine_types="ecs.gn7i-c32g1.8xlarge"
    else
        used_machine_types="ecs.gn6e-c12g1.12xlarge"
    fi
else
    used_machine_types="ecs.gn7i-c32g1.32xlarge,ecs.gn6e-c12g1.12xlarge"
fi

mkdir -p /root/logs
log_path="/root/logs/expr_${expr}_${exec_type}_${model_set}_${backend}_${cv}_${req_rate}_main.log"

vllm() {
    cd $cur_dir/../scripts/kubernetes/vllm
    SLOW_EXPR=1 python src/start_storage_server.py
    SLOW_EXPR=1 USED_MACHINES=$used_machine_types python src/main.py > $log_path 2>&1 &
}

serverlessllm() {
    if [ "$expr" == "0" ]; then
        export EXPR_1_1=1
    else
        # Get model list
        python src/request_generator.py trace/trace_${cv}.pkl $req_rate model_${cv}_${req_rate}.txt
    fi
    cd $cur_dir/../scripts/kubernetes/serverlesslllm
    python src/init_servers.py
    pod_ip=$(cat head_ip.txt)
    export SERVER_POD_IP=$pod_ip
    export LLM_SERVER_URL=http://${SERVER_POD_IP}:8343/
    python src/deploy_models.py model_${cv}_${req_rate}.txt
}

hydraserve_with_single_worker() {
    python src/start_storage_server.py
    MAX_PP_SIZE=1 USED_MACHINES=$used_machine_types python src/main.py > $log_path 2>&1 &
}

hydraserve() {
    python src/start_storage_server.py
    USED_MACHINES=$used_machine_types python src/main.py > $log_path 2>&1 &
}

hydraserve_with_cache() {
    USE_CACHE=1 python src/start_storage_server.py
    USED_MACHINES=$used_machine_types USE_CACHE=1 python src/main.py > $log_path 2>&1 &
}

export MODEL_SET=$model_set
case "$exec_type" in
    "serverless_vllm")
        vllm
        ;;
    "serverlessllm")
        serverlessllm
        ;;
    "serverlessllm_with_cached_model")
        serverlessllm
        ;;
    "hydraserve_with_single_worker")
        hydraserve_with_single_worker
        ;;
    "hydraserve")
        hydraserve
        ;;
    "hydraserve_with_cache")
        hydraserve_with_cache
        ;;
    *)
        echo "unrecognized exec_type: $exec_type"
        exit 1
        ;;
esac

echo "Endpoint started. Log printed to ${log_path}."
echo "Waiting for Endpoint startup..."
sleep 10
echo "Server Started."
