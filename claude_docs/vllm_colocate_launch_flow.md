# On-Policy Distillation: vLLM Student/Teacher 서버 기동 상세 분석 (Colocate 모드)

## 개요

On-policy distillation에서 student와 teacher 모델은 각각 vLLM `AsyncLLM` 엔진으로 기동된다.  
**Colocate 모드**란 teacher가 별도 GPU 자원을 갖지 않고, student(actor)와 **동일한 Ray placement group** 안에서 GPU를 공유하는 방식이다.

---

## 1. Student vs Teacher vLLM 서버 비교

| 항목 | Student | Teacher |
|---|---|---|
| `load_format` | `dummy` (HYBRID) / `auto` (COLOCATED) | `auto` |
| `rollout_mode` | `HYBRID` (training worker GPU 공유) | `COLOCATED` |
| 가중치 갱신 | 매 step마다 ZMQ/SHM IPC 동기화 | 고정 (학습 안 함) |
| Ray actor 이름 | `server_{replica}_{node}` | `server_teacher_{replica}_{node}` |
| Config 경로 | `actor_rollout_ref.rollout.*` | `distillation.teacher_model.inference.*` |
| `is_teacher_model` 플래그 | `False` | `True` |
| Resource Pool | `global_pool` (actor와 동일) | `global_pool` split (colocate 시) |
| `max_logprobs` | 미설정 | `distillation_loss.topk` 자동 주입 |

---

## 2. Colocate 모드 활성화 조건

### config 설정

```bash
# run_qwen_gsm8k.sh:78
distillation.teacher_model.enable_resource_pool=False   # False = colocate
```

```yaml
# verl/trainer/config/distillation/distillation.yaml:66
enable_resource_pool: false
```

### Resource Pool 초기화 (`main_ppo.py:224-287`)

```python
if distillation_config.teacher_model.enable_resource_pool:
    # Standalone: teacher 전용 "teacher_pool" 생성
    resource_pool_spec["teacher_pool"] = [n_gpus_per_node] * nnodes
    mapping[Role.TeacherModel] = "teacher_pool"
else:
    # Colocate: teacher가 student와 동일한 "global_pool" 공유
    distillation_config.teacher_model.nnodes = config.trainer.nnodes
    distillation_config.teacher_model.n_gpus_per_node = config.trainer.n_gpus_per_node
    mapping[Role.TeacherModel] = "global_pool"
```

---

## 3. Student vLLM 실행 경로 (Colocate)

### 전체 호출 스택

```
main_ppo.py
  └─ TaskRunner.run()
       └─ RayPPOTrainer.__init__()   [ray_trainer.py]
            └─ AgentLoopManager 생성
                 └─ ActorRolloutRefWorker.init_model()   [engine_workers.py:493~622]
                      └─ get_rollout_class("vllm", "async") → ServerAdapter
                           └─ ServerAdapter.__init__()   [vllm_rollout.py:61~99]
                                └─ (ZMQ handle로 vLLM 서버와 통신)
```

### Step 1 — rollout 클래스 선택 (`engine_workers.py:601-604`)

```python
rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
rollout_cls = get_rollout_class(rollout_config.name, rollout_config.mode)
# name="vllm", mode="async" → ServerAdapter 반환
self.rollout = rollout_cls(
    config=rollout_config,
    model_config=model_config,
    device_mesh=rollout_device_mesh,
)
```

### Step 2 — AgentLoopManager에서 vLLMReplica 생성 (`agent_loop.py:1056-1090`)

```python
self.rollout_replicas = [
    vLLMReplica(
        replica_rank=rank,
        config=self.rollout_config,      # actor_rollout_ref.rollout
        model_config=self.model_config,
        gpus_per_node=...,
        is_teacher_model=False,          # student
    )
    for rank in range(num_replicas)
]

# training worker_group이 있으면 → HYBRID 모드 (student 기본)
if self.worker_group:
    await server.init_hybrid(self.worker_group)
else:
    await server.init_standalone()
```

### Step 3 — `init_hybrid()`: training worker GPU 공유 (`replica.py:144-154`)

```python
async def init_hybrid(self, worker_group: RayWorkerGroup):
    self.rollout_mode = RolloutMode.HYBRID
    # training worker들 중 이 replica에 해당하는 슬라이스 사용
    self.workers = worker_group.workers[
        self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
    ]
    await self.launch_servers()   # → vLLMHttpServer Ray actor 생성
```

---

## 4. Teacher vLLM 실행 경로 (Colocate)

### 전체 호출 스택

```
ray_trainer.py:844~851
  └─ TeacherModelManager(config, resource_pool=global_pool)   [teacher_model.py]
       └─ _initialize_llm_servers()
            └─ vLLMReplica(replica_rank, ..., is_teacher_model=True)
                 └─ server.init_colocated(split_resource_pool)   ← colocate 핵심 분기
                      └─ RayWorkerGroup(resource_pool=split_pool)
                           └─ launch_servers()
```

### Step 1 — TeacherModelManager 생성 (`ray_trainer.py:844-851`)

```python
if self.use_teacher_policy:
    from verl.experimental.teacher_loop import TeacherModelManager

    teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
    # colocate 시: teacher_resource_pool = global_pool (student와 동일)
    self.teacher_model_manager = TeacherModelManager(
        config=self.config.distillation,
        resource_pool=teacher_resource_pool,   # None이 아님 → colocate 경로
    )
```

### Step 2 — `_initialize_llm_servers()` 핵심 분기 (`teacher_model.py:56-98`)

```python
def _initialize_llm_servers(self):
    teacher_world_size = (
        teacher_model_config.inference.tensor_model_parallel_size
        * teacher_model_config.inference.data_parallel_size
        * teacher_model_config.inference.pipeline_model_parallel_size
    )
    # Colocate: resource_pool.world_size 사용 (standalone은 n_gpus_per_node * nnodes)
    world_size = (
        self.resource_pool.world_size   # colocate 경로
        if self.resource_pool
        else teacher_model_config.n_gpus_per_node * teacher_model_config.nnodes
    )
    num_replicas = world_size // teacher_world_size

    self.rollout_replicas = [
        rollout_replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=teacher_model_config.n_gpus_per_node,
            is_teacher_model=True,
        )
        for replica_rank in range(num_replicas)
    ]

    # 핵심 분기: resource_pool 유무
    if self.resource_pool:
        # Colocate: global_pool을 teacher_world_size 단위로 분할
        split_resource_pools = split_resource_pool(self.resource_pool, teacher_world_size)
        self._run_all([
            server.init_colocated(resource_pool)
            for server, resource_pool in zip(self.rollout_replicas, split_resource_pools)
        ])
    else:
        # Standalone
        self._run_all([server.init_standalone() for server in self.rollout_replicas])
```

### Step 3 — `init_colocated()`: actor와 동일한 placement group 사용 (`replica.py:173-200`)

```python
async def init_colocated(self, resource_pool: RayResourcePool):
    self.rollout_mode = RolloutMode.COLOCATED
    self.resource_pool = resource_pool

    name_prefix = (
        f"rollout_teacher_colocate_{self.replica_rank}"
        if self.is_teacher_model
        else f"rollout_colocate_{self.replica_rank}"
    )

    # 핵심: actor와 동일한 resource_pool(placement group) 사용
    worker_group = RayWorkerGroup(
        resource_pool=self.resource_pool,
        ray_cls_with_init=self.get_ray_class_with_init_args(),
        bin_pack=False,
        name_prefix=name_prefix,
        use_gpu=use_gpu,
    )
    self.workers = worker_group.workers
    await self.launch_servers()
```

---

## 5. `launch_servers()`: vLLMHttpServer Ray actor 생성 (`vllm_async_server.py:881-959`)

Student, Teacher 공통 경로. `is_teacher_model` 플래그로 이름만 구분.

```python
async def launch_servers(self):
    # 1. 모든 worker에서 node_id와 CUDA device 번호 수집
    worker_infos = await asyncio.gather(*[
        worker.__ray_call__.remote(lambda self: (
            ray.get_runtime_context().get_node_id(),
            ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
        ))
        for worker in self.workers
    ])
    worker_node_ids = [info[0] for info in worker_infos]
    worker_cuda_visible_devices = [info[1] for info in worker_infos]

    # 2. 노드별로 vLLMHttpServer Ray actor 생성
    for node_rank in range(nnodes):
        node_cuda_visible_devices = ",".join(
            worker_cuda_visible_devices[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
        )
        node_id = worker_node_ids[node_rank * gpus_per_node]

        # teacher/student 이름 구분
        name = (
            f"server_teacher_{self.replica_rank}_{node_rank}"
            if self.is_teacher_model
            else f"server_{self.replica_rank}_{node_rank}"
        )

        server = vLLMHttpServer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,    # actor와 동일 노드에 고정
                soft=False,
            ),
            runtime_env={"env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",   # Ray가 덮어쓰지 못하게 차단
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }},
            name=name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=node_rank,
            gpus_per_node=gpus_per_replica_node,
            nnodes=nnodes,
            cuda_visible_devices=node_cuda_visible_devices,   # actor worker의 실제 GPU 번호
        )
        self.servers.append(server)

    # 3. 모든 서버에 launch_server() 호출 (master_address/port 공유)
    master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
    await asyncio.gather(*[
        server.launch_server.remote(
            master_address=master_address,
            master_port=master_port,
            dp_rpc_port=dp_rpc_port,
        )
        for server in self.servers
    ])
```

> **핵심**: `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`로 Ray의 자동 GPU 격리를 막고,  
> actor 워커에서 읽어온 실제 GPU device 번호를 그대로 `cuda_visible_devices`로 주입한다.  
> 이를 통해 teacher/student가 **같은 물리적 GPU**를 별도 프로세스에서 공유한다.

---

## 6. `launch_server()`: vLLM AsyncEngineArgs 구성 (`vllm_async_server.py:193-372`)

### Config에서 오는 파라미터

```python
# vllm_async_server.py:241-266
args = {
    # ---- 공통 필드 ----
    "dtype":                    self.config.dtype,                    # "auto" / "bfloat16"
    "load_format":              self.config.load_format,              # student: "dummy", teacher: "auto"
    "skip_tokenizer_init":      False,
    "distributed_executor_backend": "mp",
    "worker_extension_cls":     self._get_worker_extension_cls(),     # vLLMColocateWorkerExtension 등
    "trust_remote_code":        self.model_config.trust_remote_code,
    "max_model_len":            self.config.max_model_len,
    "max_num_seqs":             self.config.max_num_seqs,
    "enable_chunked_prefill":   self.config.enable_chunked_prefill,
    "max_num_batched_tokens":   self.config.max_num_batched_tokens,
    "enable_prefix_caching":    self.config.enable_prefix_caching,
    "enable_sleep_mode":        self.config.enable_sleep_mode,        # True → training 중 GPU 양보
    "logprobs_mode":            self.config.logprobs_mode,            # "processed_logprobs"
    "enforce_eager":            self.config.enforce_eager,
    "gpu_memory_utilization":   self.config.gpu_memory_utilization,
    "disable_log_stats":        self.config.disable_log_stats,
    "tensor_parallel_size":     self.config.tensor_model_parallel_size,
    "scheduling_policy":        self.config.scheduling_policy,
    **engine_kwargs,            # config.engine_kwargs.vllm.* 추가 오버라이드
}
```

### 런타임에 계산되는 파라미터

```python
args["seed"]         = self.replica_rank + self.config.get("seed", 0)
args["model"]        = self.model_config.local_path    # HF 모델 경로

# 포트는 node_rank==0에서 get_free_port()로 동적 할당
self._master_port    = get_free_port(self._server_address)
self._dp_rpc_port    = get_free_port(self._server_address)
self._dp_master_port = get_free_port(self._server_address)
```

### Data Parallel 파라미터 (DP > 1인 경우)

```python
if self.config.data_parallel_size > 1:
    data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
    args.update({
        "data_parallel_size":        self.config.data_parallel_size,
        "data_parallel_size_local":  data_parallel_size_local,
        "data_parallel_start_rank":  self.node_rank * data_parallel_size_local,
        "data_parallel_address":     self._master_address,
        "data_parallel_rpc_port":    self._dp_rpc_port,
    })
```

### Multi-node 파라미터 (nnodes > 1인 경우)

```python
if self.nnodes > 1:
    args.update({
        "master_addr":          self._master_address,
        "master_port":          self._master_port,
        "node_rank":            self.node_rank,
        "nnodes":               self.nnodes,
        "data_parallel_address":  self._master_address,
        "data_parallel_rpc_port": self._dp_rpc_port,
    })
```

### Teacher 전용 — `max_logprobs` 자동 주입 (`distillation.py:179-196`)

```python
# distillation_loss.topk 설정 시 teacher vLLM engine_kwargs에 자동 추가
if engine_name == "vllm":
    vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
    if vllm_engine_kwargs.get("max_logprobs") is None:
        vllm_engine_kwargs["max_logprobs"] = self.distillation_loss.topk  # 예: 64
    engine_kwargs["vllm"] = vllm_engine_kwargs
```

---

## 7. AsyncLLM 최종 생성 (`vllm_async_server.py:374-414`)

Student, Teacher 모두 동일한 코드 경로:

```python
async def run_server(self, args: argparse.Namespace):
    # CLI args → AsyncEngineArgs → vllm_config
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

    # 실제 vLLM 엔진 생성
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        **kwargs,
    )

    # HTTP 서버(OpenAI API compatible) 기동
    app = build_app(args)
    await init_app_state(engine_client, vllm_config, app.state, args)
    self.engine = engine_client
    self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)
```

---

## 8. Config 파라미터 → vLLM 인자 매핑 요약

### Student (`actor_rollout_ref.rollout.*`)

| Shell 변수 / Config 경로 | vLLM AsyncEngineArgs 필드 | 예시값 |
|---|---|---|
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `tensor_parallel_size` | 1 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `gpu_memory_utilization` | 0.3 |
| `actor_rollout_ref.rollout.max_model_len` | `max_model_len` | 769 |
| `actor_rollout_ref.rollout.max_num_seqs` | `max_num_seqs` | 769 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `max_num_batched_tokens` | 769 |
| `actor_rollout_ref.rollout.enforce_eager` | `enforce_eager` | True |
| `actor_rollout_ref.rollout.dtype` | `dtype` | "auto" |
| `actor_rollout_ref.rollout.enable_sleep_mode` | `enable_sleep_mode` | True |
| (하드코딩) | `load_format` | "dummy" (HYBRID 모드) |
| (하드코딩) | `distributed_executor_backend` | "mp" |
| (런타임) `model_config.local_path` | `model` | "/path/to/student" |

### Teacher (`distillation.teacher_model.inference.*`)

| Shell 변수 / Config 경로 | vLLM AsyncEngineArgs 필드 | 예시값 |
|---|---|---|
| `distillation.teacher_model.inference.tensor_model_parallel_size` | `tensor_parallel_size` | 1 |
| `distillation.teacher_model.inference.gpu_memory_utilization` | `gpu_memory_utilization` | 0.3 |
| `distillation.teacher_model.inference.max_model_len` | `max_model_len` | 769 |
| `distillation.teacher_model.inference.max_num_seqs` | `max_num_seqs` | 769 |
| `distillation.teacher_model.inference.max_num_batched_tokens` | `max_num_batched_tokens` | 769 |
| `distillation.teacher_model.inference.enforce_eager` | `enforce_eager` | True |
| `distillation.teacher_model.model_path` | `model` | "Qwen/Qwen2.5-3B-Instruct" |
| `distillation.distillation_loss.topk` | `max_logprobs` (engine_kwargs 자동 주입) | 64 |
| (하드코딩) | `load_format` | "auto" |
| (런타임) `get_free_port()` | `master_port`, `data_parallel_rpc_port` | 동적 할당 |

---

## 9. Colocate vs Standalone 비교

| 항목 | Colocate 모드 | Standalone 모드 |
|---|---|---|
| Config 플래그 | `enable_resource_pool=False` | `enable_resource_pool=True` |
| Resource Pool | `global_pool` 공유 | 별도 `teacher_pool` |
| 초기화 메서드 | `init_colocated(resource_pool)` | `init_standalone()` |
| GPU 배정 | actor worker에서 동적으로 읽어옴 | 전용 pool에서 할당 |
| CUDA_VISIBLE_DEVICES | actor 워커 GPU 번호 그대로 | 독립적으로 할당 |
| Ray placement group | actor와 동일 | 별도 생성 |
| Role 매핑 | `TeacherModel → "global_pool"` | `TeacherModel → "teacher_pool"` |
| `rollout_mode` | `COLOCATED` | `STANDALONE` |

---

## 10. 관련 핵심 파일 위치

| 파일 | 역할 |
|---|---|
| `verl/trainer/main_ppo.py` | resource pool 초기화, role→pool 매핑 |
| `verl/trainer/ppo/ray_trainer.py` | `TeacherModelManager` 생성 (line 844) |
| `verl/experimental/teacher_loop/teacher_model.py` | `TeacherModelManager._initialize_llm_servers()` |
| `verl/workers/rollout/replica.py` | `init_hybrid / init_colocated / init_standalone` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `vLLMReplica.launch_servers()`, `vLLMHttpServer.launch_server()`, `run_server()` |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `ServerAdapter` (student rollout 클라이언트) |
| `verl/workers/engine_workers.py` | `ActorRolloutRefWorker.init_model()` (line 601) |
| `verl/workers/config/distillation.py` | `max_logprobs` 자동 주입 (line 179) |
| `verl/trainer/config/distillation/distillation.yaml` | teacher 기본 config |
| `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopManager._initialize_llm_servers()` |
