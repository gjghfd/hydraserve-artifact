# HydraServe Artifact

This is the artifact for the paper "HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds".
This guide provides instructions to reproduce the main results presented in the paper.

[NOTE: end-to-end scripts have not been tested.]

## Environment Setup

We have prepared the environment on Aliyun ACK cluster. Please refer to HotCRP to obtain the access credentials for the cluster.

To set up environment on your own servers, please refer to the installation sections in [HydraServe Setup Guide](Installation.md) and [ServerlessLLM Setup Guide](../scripts/kubernetes/serverlessllm/README.md).

Due to the budget limit, we can only provision the cluster for a short time slot. Thus, we only reproduce the main results in our paper (figure7,9,11,13).

### GPU Isolation Configuration
HydraServe and ServerlessLLM require different GPU isolation strategies. Before switching from one system to the other for experiments, please make sure to change the GPU isolation strategy accordingly (Serverless vLLM uses the same strategy as HydraServe).
To configure the GPU isolation strategy, you should change the labels of all GPU servers:

1. For HydraServe, run
```
cd hydraserve-artifact/scripts/kubernetes
SHARE=1 python label_nodes.py
```
2. For ServerlessLLM, run
```
cd hydraserve-artifact/scripts/kubernetes
SHARE=0 python label_nodes.py
```

## Figure 7 (Cold Start Latency)

As the remote storage cannot concurrently supply too many models, we have split the models into two sets and will run the experiments twice.

For each execution type (`serverless_vllm, serverlessllm, serverlessllm_with_cached_model, hydraserve_with_single_worker, hydraserve`), each model set (`0, 1`), and each backend (`a10, v100`), first start the server by
```
export exec_type=[execution_type]
export model_set=[model_set]
export backend=[backend]
sh ./start_server.sh 0 $exec_type $model_set $backend 0 0
```

Then, run the cold start experiment:
```
sh ./coldstart.sh $exec_type $model_set $backend
```

After the experiments for all settings have completed, use `figure7.py` to generate the figure `figs/figure7.pdf`.

## Figure 9 (End-to-End Performance)

NOTE: Due to time limits, you can choose several settings out of all settings to run end-to-end experiment. The settings that you did not run will be replaced with results in our experiments. We suggest you to prioritize experiments under CV=8 and req_rate=0.6 to successfully generate Figure 11 and Figure 13 in the paper.

For each execution type (`serverless_vllm, serverlessllm, hydraserve, hydraserve_with_cache`), each CV (`8,4,2`), and each request rate (`0.6, 0.7, 0.8`), first start the server by
```
export exec_type=[execution_type]
export cv=[cv]
export req_rate=[request_rate]
sh ./start_server.sh 1 ${exec_type} 3 hybrid ${cv} ${req_rate}
```

Then, run the end-to-end experiment:
```
sh ./end2end.sh $exec_type $cv $req_rate
```
The experiment will elapse an hour.

After the experiments for your selected settings have been completed, use `figure9.py` to generate the figure `figs/figure9.pdf`.

## Figure 11 (Application Analysis)

After obtaining all the results of the end-to-end experiment under CV=8 and req_rate=0.6, use `figure11.py` to generate the figure `figs/figure11.pdf`. 

## Figure 13 (TPOT and Resource Usage Penalties)

After obtaining all the results of the end-to-end experiment under CV=8 and req_rate=0.6, use `figure13.py` to generate the figures `figs/figure13-a.pdf` and `figs/figure13-b.pdf`. 