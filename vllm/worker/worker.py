"""A GPU worker class."""
import gc
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              get_tensor_model_parallel_cpu_group,
                              init_distributed_environment)
from vllm.distributed.device_communicators import pynccl_utils
from vllm.distributed.device_communicators.custom_all_reduce import (
    init_custom_ar)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker_base import WorkerBase


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        self.gpu_cache: List[torch.Tensor]

        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))
        self.dest_pp_size = int(os.getenv('DEST_PP_SIZE', '1'))
        self.dest_pp_rank = -1
        if self.dest_pp_size < self.pp_size:
            self.dest_pp_rank = int(os.getenv('DEST_PP_RANK', '-1'))

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            if os.getenv("LOCAL_INFERENCE", "0") == "1":
                rank = int(os.getenv("PP_RANK", "0"))
                self.device = torch.device(f"cuda:{rank}")
            else:
                self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
            print(f"device id = {torch.cuda.current_device()}")

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        if int(os.getenv("FAST_COLDSTART", "1")) == 0:
            self.model_runner.profile_run()
            torch.cuda.synchronize()
            free_memory = torch.cuda.mem_get_info()[0]
        else:
            # Execute a forward pass with dummy inputs to profile the memory usage
            # of the model.
            print("Faster Cold-start: disable profile_run and calculate activation memory usage.")
            print("Warning: assume we are using float16 or bfloat16.")

            hidden_size = self.model_config.get_hidden_size()
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            # batch_size = int(os.getenv("BATCH_SIZE", "8"))
            # max_tokens = int(os.getenv("MAX_TOKENS", 1024))
            # activation_size = batch_size * max_tokens * hidden_size * num_layers * 2
            free_gpu_memory = torch.cuda.mem_get_info()[0]

            print(f"Total freed gpu memory = {free_gpu_memory / float(2**30)} GB")

            # The final free_memory Satisfy:
            # 1. Actication_Size = free_memory / cache_block_size * self.cache_config.block_size * hidden_size * num_layers * 2
            # 2. Actication_Size + free_memory <= free_gpu_memory

            cache_block_size = self.get_cache_block_size_bytes()
            free_memory = free_gpu_memory / (1 + 1.0 / cache_block_size * self.cache_config.block_size * hidden_size * num_layers * 2)

        print(f"free_gpu_memory = {free_memory / float(2**30)} GB, target utilization = {self.cache_config.gpu_memory_utilization}")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (free_memory * self.cache_config.gpu_memory_utilization) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        print(f"determine_num_available_blocks: available num gpu blocks = {num_gpu_blocks}")

        cur_pp_size = int(os.getenv('PP_SIZE', '1'))
        if self.dest_pp_size < cur_pp_size:
            # only allocate few gpu blocks for cold start worker when we will load dest model later
            if int(os.getenv("FAST_COLDSTART", "1")) == 0:
                max_num_gpu_block = 512
            else:
                dest_model_scale = cur_pp_size / self.dest_pp_size

                # torch.cuda.mem_get_info is not true in Aliyun cGPU so we need to use total_gpu_mem from environment
                total_gpu_mem = int(os.getenv("TOTAL_GPU", "0")) * 1024 * 1024 * 1024
                if total_gpu_mem == 0:
                    total_gpu_mem = torch.cuda.mem_get_info()[1]

                hidden_size = self.model_config.get_hidden_size()
                num_layers = self.model_config.get_num_layers(self.parallel_config)

                predict_free_memory_dest = total_gpu_mem - (total_gpu_mem - free_gpu_memory) * dest_model_scale
                # Satisfy: 
                # 1. activation_memory_dest = max_num_gpu_block * self.cache_config.block_size * hidden_size * num_layers * dest_model_scale * 2
                # 2. (predict_free_memory_dest - activation_memory_dest - max_num_gpu_block * cache_block_size) * gpu_memory_utilization > max_num_gpu_block * cache_block_size * (dest_model_scale - 1)
                max_num_gpu_block = int(predict_free_memory_dest / ((self.cache_config.block_size * hidden_size * num_layers * dest_model_scale * 2 + cache_block_size) * self.cache_config.gpu_memory_utilization + cache_block_size * (dest_model_scale - 1)))
            num_gpu_blocks = min(num_gpu_blocks, max_num_gpu_block)
        elif self.cache_config.num_gpu_blocks is not None:
            # we are loading dest model
            last_num_gpu_blocks = self.cache_config.num_gpu_blocks
            if last_num_gpu_blocks > num_gpu_blocks:
                print(f"Warning: loading dest model and previous gpu blocks is larger ({last_num_gpu_blocks}), set to previous one")
                num_gpu_blocks = last_num_gpu_blocks
        
        print(f"determine_num_available_blocks: get num_gpu_blocks = {num_gpu_blocks}")

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        use_dest: Optional[bool] = False,
    ) -> None:
        # Issue cache operations.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in, use_dest)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out, use_dest)
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy, use_dest)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:

        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            assert execute_model_req is not None
            num_seq_groups = len(seq_group_metadata_list)
            blocks_to_swap_in = execute_model_req.blocks_to_swap_in
            blocks_to_swap_out = execute_model_req.blocks_to_swap_out
            blocks_to_copy = execute_model_req.blocks_to_copy
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        use_dest = execute_model_req.use_dest
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy, use_dest)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, use_dest)

        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # NOTE(woosuk): We don't initialize pynccl process group when world size
        # is 1.
        # NOTE(kaichao): By default, pynccl is initialized for tp group.
        pynccl_utils.init_process_group(
            group=get_tensor_model_parallel_cpu_group())

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()

    # Note(chiheng): No warmup.
    # A small all_reduce for warmup.
    # torch.distributed.all_reduce(torch.zeros(1).cuda())
    # if pynccl_utils.is_initialized():
    #     pynccl_utils.all_reduce(torch.zeros(1).cuda())


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
