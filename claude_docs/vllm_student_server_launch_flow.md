# verl vllm Student 서버 기동 및 데이터 흐름 상세 분석

## 개요

On-policy KD 또는 일반 PPO/GRPO 학습 시 student 모델은 vllm `AsyncLLM` 엔진으로 기동된다.
Teacher와 달리 **HYBRID mode** (training worker와 GPU 공유) 로 동작하며, 가중치를 파일에서 읽지 않고 매 step마다 training worker로부터 IPC로 동기화받는 것이 핵심이다.

---

## 1. Student vs Teacher vllm 서버 비교

| 항목 | Student | Teacher |
|---|---|---|
| `load_format` | `dummy` (가중치 없이 엔진만 기동) | `auto` (checkpoint에서 직접 로드) |
| `rollout_mode` | `HYBRID` (training worker GPU 공유) | `COLOCATED` or `STANDALONE` |
| `worker_extension_cls` | `vLLMColocateWorkerExtension` | 동일 |
| 가중치 갱신 | 매 step마다 ZMQ/SHM IPC로 동기화 | 고정 (학습 안 함) |
| Ray actor 이름 | `vllm_server_{replica}_{node}` | `vllm_server_teacher_{replica}_{node}` |
| 설정 경로 | `config.actor_rollout_ref.rollout` | `config.distillation.teacher_model.inference` |
| HTTP server | 있음 (node_rank==0) | 있음 (node_rank==0) |

---

## 2. Student vllm 서버 기동 전체 흐름

### Step 1 — config `load_format: dummy` (`rollout/rollout.yaml:92`)

```yaml
# verl/trainer/config/rollout/rollout.yaml
load_format: dummy
```

`dummy`는 "가중치를 파일에서 읽지 말고, training worker가 나중에 직접 밀어넣겠다"는 의미.
단, HYBRID가 아닌 mode에서 `dummy`이면 `auto`로 강제 변경됨:

```python
# vllm_async_server.py:126-128
if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
    self.config.load_format = "auto"
```

---

### Step 2 — `AgentLoopManager.create()` → `_initialize_llm_servers()` (`agent_loop.py:1056`)

```python
# agent_loop.py:1069-1090
self.rollout_replicas = [
    vLLMReplica(
        replica_rank=rank,
        config=self.rollout_config,      # actor_rollout_ref.rollout
        model_config=self.model_config,  # actor_rollout_ref.model
        gpus_per_node=...,
        is_teacher_model=False,          # student
    )
    for rank in range(num_replicas)
]

if self.worker_group:   # training worker_group이 있으면 → hybrid 모드
    await server.init_hybrid(self.worker_group)
else:
    await server.init_standalone()
```

---

### Step 3 — `init_hybrid()`: training worker GPU 공유 (`replica.py:144`)

```python
# replica.py:144-154
async def init_hybrid(self, worker_group: RayWorkerGroup):
    self.rollout_mode = RolloutMode.HYBRID
    # training worker들 중 이 replica에 해당하는 슬라이스 사용
    self.workers = worker_group.workers[
        self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
    ]
    await self.launch_servers()   # → vLLMHttpServer Ray actor 생성
```

---

### Step 4 — `launch_servers()`: `vLLMHttpServer` Ray actor 생성 (`vllm_async_server.py:881`)

```python
# 각 training worker에서 node_id와 GPU id 수집
worker_infos = await asyncio.gather(
    *[worker.__ray_call__.remote(lambda self: (node_id, gpu_id)) for worker in self.workers]
)

# 노드별로 vLLMHttpServer Ray actor를 같은 노드에 NodeAffinity로 배치
for node_rank in range(nnodes):
    server = ray.remote(vLLMHttpServer).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id),
        name=f"vllm_server_{replica_rank}_{node_rank}",
        max_concurrency=max_num_seqs + 16,
        runtime_env={"env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
        }},
    ).remote(
        config=rollout_config,
        model_config=model_config,
        rollout_mode=RolloutMode.HYBRID,
        workers=workers,                      # training worker 핸들
        cuda_visible_devices="0,1,2,3",       # training worker와 동일한 GPU
    )

# 모든 서버에 launch_server() 호출
master_address, master_port, _ = await self.servers[0].get_master_address.remote()
await asyncio.gather(*[
    server.launch_server.remote(master_address=master_address, master_port=master_port)
    for server in self.servers
])

# _server_handle, _server_address 세팅
self._server_handle  = self.servers[0]          # generate() 시 쓰이는 Ray actor handle
self._server_address = f"{ip}:{port}"
```

---

### Step 5 — `launch_server()`: vllm CLI args 조립 → `AsyncLLM` 기동 (`vllm_async_server.py:193`)

```python
args = {
    "load_format":            "dummy",   # 가중치 없이 기동
    "worker_extension_cls":   "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    "dtype":                  config.dtype,
    "tensor_parallel_size":   config.tensor_model_parallel_size,
    "gpu_memory_utilization": config.gpu_memory_utilization,
    "max_model_len":          config.max_model_len,
    "max_num_seqs":           config.max_num_seqs,
    "enable_chunked_prefill": config.enable_chunked_prefill,
    "enable_prefix_caching":  config.enable_prefix_caching,
    "enable_sleep_mode":      config.enable_sleep_mode,  # True → training 중 GPU 양보
    "enforce_eager":          config.enforce_eager,
    "disable_log_stats":      config.disable_log_stats,
    "seed":                   replica_rank + config.seed,
    **config.engine_kwargs.get("vllm", {}),              # 추가 vllm 전용 옵션
}

# dict → CLI 문자열 리스트 변환
server_args = ["serve", model_config.local_path] + build_cli_args_from_config(args)
# 예: ["serve", "/path/to/student", "--load-format", "dummy",
#      "--tensor-parallel-size", "4", "--gpu-memory-utilization", "0.4", ...]

# CLI args → AsyncEngineArgs → vllm_config → AsyncLLM 기동
engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
vllm_config = engine_args.create_engine_config()
self.engine = AsyncLLM.from_vllm_config(vllm_config)   # 엔진 기동 (가중치 비어있음)

# monkey_patch: OOV token sampling 방지
await self.engine.collective_rpc("monkey_patch_model", kwargs={"vocab_size": tokenizer_vocab_size})

# HTTP server (OpenAI API compatible) 기동
self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)
```

---

### Step 6 — 매 training step마다 student 가중치 동기화

training 후 `ServerAdapter.update_weights()` 호출 시:

```
training worker (FSDP/Megatron)
  → BucketedWeightSender (ZMQ 또는 SHM)
    → vLLMHttpServer.engine.collective_rpc("update_weights_from_ipc")
      → vLLMColocateWorkerExtension.update_weights_from_ipc()  (utils.py:179)
        → BucketedWeightReceiver.receive_weights()
          → model_runner.model에 직접 tensor 덮어씌움
          → process_weights_after_loading()
      → engine.wake_up(tags=["kv_cache", "weights"])  ← rollout 준비 완료
```

`vLLMColocateWorkerExtension` (`utils.py:122`)은 `worker_extension_cls`로 vllm 내부 worker process에 주입된 클래스로, weight 동기화 메서드(`update_weights_from_ipc`, `_update_weights`)를 제공한다.

---

## 3. generate 시 server 사용 흐름

```
SingleTurnAgentLoop.run()                      (single_turn_agent_loop.py:60)
  └─ self.server_manager.generate(prompt_ids)  (AsyncLLMServerManager)
       └─ _acquire_server(request_id)          ← GlobalRequestLoadBalancer (LFU)
            └─ server.generate.remote(...)     ← vLLMHttpServer Ray actor
                 └─ vllm AsyncLLM.generate()   ← 실제 추론
                      └─ TokenOutput(token_ids=final_res.outputs[0].token_ids)
```

---

## 4. token_ids → batch["responses"] → decode 경로

```
vllm_async_server.py:522
  token_ids = final_res.outputs[0].token_ids         # list[int]
  ↓ TokenOutput(token_ids=...)
single_turn_agent_loop.py:73
  AgentLoopOutput(response_ids=output.token_ids[:response_length])
  ↓ _agent_loop_postprocess()  (agent_loop.py:615)
  _InternalAgentLoopOutput(response_ids=torch.Tensor)
  ↓ _postprocess()  (agent_loop.py:870)
  TensorDict({"responses": response_ids})            # [bsz, response_length]
  ↓ DataProto(batch=...)
ray_trainer.py:444
  tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
```

`self.tokenizer`는 `main_ppo.py:351`에서 `config.actor_rollout_ref.model.path`로부터 로드한 **student tokenizer**이다.

---

## 5. 관련 핵심 파일 위치

| 파일 | 역할 |
|---|---|
| `verl/trainer/config/rollout/rollout.yaml` | student rollout config 기본값 (`load_format: dummy`) |
| `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopManager`, `AgentLoopWorker`, `AsyncLLMServerManager` |
| `verl/workers/rollout/replica.py` | `RolloutReplica` base class, `init_hybrid/standalone/colocated` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `vLLMHttpServer`, `vLLMReplica`, `launch_server()`, `generate()` |
| `verl/workers/rollout/vllm_rollout/utils.py` | `vLLMColocateWorkerExtension` (weight sync) |
| `verl/experimental/agent_loop/single_turn_agent_loop.py` | `SingleTurnAgentLoop.run()` |
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer`, `batch_decode` (dump 지점) |
| `verl/trainer/main_ppo.py` | tokenizer 로드, `RayPPOTrainer` 생성 |
