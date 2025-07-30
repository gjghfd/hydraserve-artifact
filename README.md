# HydraServe
This is an implementation of the paper: "HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds".

## Getting Started Instructions

### Code Structure

```
- This repository: Contains our custom serverless LLM serving framework, based on vLLM v0.4.2.
  - scripts/                # Top-level directory for all operational scripts.
    - image/                # Contains Dockerfiles and related assets for building the project's container images.the images.
      - kubernetes/         # All scripts and components for deploying and managing the framework on Kubernetes.
        - vllm/             # The core components of our serverless LLM serving framework.
          - src/            # The main source code for the serving framework logic.
          - download/       # A utility service to download and prepare LLM models, including its Dockerfile.
          - storage_server: # The remote storage server component, including its source code and Dockerfile.
          - trace:          # A tool to generate workload traces for performance end-to-end experiments.
        - serverlessllm:    # Scripts and configurations to run comparative benchmarks against the original ServerlessLLM framework.
        - tool-node-shell:  # Kubernetes node shell installation guide.
```

### Installation
Due to the limited time available for resource provisioning, we have prepared the environment on Aliyun ACK cluster for AE reviewers. Please refer to HotCRP to obtain the access credentials for the cluster.

To set up environment on your own servers, please follow the [Installation Guide](ae_scripts/Installation.md) to install and test HydraServe.

## Detailed Instructions

To reproduce the main results in our paper, please follow the instructions in the [Reproduce-README](ae_scripts/README.md).