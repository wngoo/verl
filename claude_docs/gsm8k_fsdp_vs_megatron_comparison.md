# GSM8K On-Policy Distillation: FSDP vs Megatron 실험 환경 비교

**대상 파일:**
- `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` (FSDP 버전)
- `examples/on_policy_distillation_trainer/run_qwen_gsmk8_megatron.sh` (Megatron 버전)

---

## 공통 설정

| 항목 | 값 |
|---|---|
| Student 모델 | `Qwen2.5-0.5B` |
| Teacher 모델 | `Qwen2.5-3B-Instruct` |
| 데이터셋 | GSM8K (train/test parquet) |
| RL 알고리즘 | GRPO (`algorithm.adv_estimator=grpo`) |
| Rollout 엔진 | vLLM (기본값, sglang으로 교체 가능) |
| Train batch size | 128 |
| Max prompt 길이 | 256 토큰 |
| Max response 길이 | 512 토큰 |
| 학습률 | `1e-6` |
| 총 에포크 | 15 |
| KL penalty | 비활성화 (`use_kl_in_reward=False`) |
| Distillation topk | 64 |

---

## 핵심 차이점

### 1. 분산 학습 백엔드 및 Config

| 항목 | FSDP | Megatron |
|---|---|---|
| **Hydra Config** | `ppo_trainer.yaml` | `ppo_megatron_trainer.yaml` |
| **백엔드** | PyTorch FSDP | Megatron-LM |
| **Student GPU 수** | 2 | 4 |
| **Teacher GPU 수** | 4 | 2 |
| **실험 이름 prefix** | `fsdp/...` | `megatron/...` |

---

### 2. 병렬화 전략

**FSDP 버전** — Ulysses Sequence Parallelism만 사용:
```bash
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
```

**Megatron 버전** — 다차원 병렬화를 세분화하여 제어:
```bash
TP=2   # Tensor Model Parallel
PP=1   # Pipeline Model Parallel
CP=1   # Context Parallel
EP=1   # Expert Model Parallel
ETP=1  # Expert Tensor Parallel
```

Megatron은 TP=2로 텐서 병렬화를 활성화해, 단일 레이어의 가중치를 2개 GPU에 분산 배치합니다.

---

### 3. 증류 손실 함수 및 학습 방식

| 항목 | FSDP | Megatron |
|---|---|---|
| `DISTILLATION_LOSS_MODE` | `k1` | `forward_kl_topk` |
| `USE_POLICY_GRADIENT` | `True` | `False` |

- **k1 (FSDP)**: Reverse KL의 단순화 변형. Policy Gradient 손실과 혼합하여 RL 탐색 효과를 포함한 증류 수행.
- **forward_kl_topk (Megatron)**: Teacher 토큰 분포 상위 K개(topk=64)에 대한 Forward KL. Policy Gradient 없이 순수하게 Teacher 분포를 학습.

> FSDP 버전은 RL + 증류 혼합 학습, Megatron 버전은 순수 증류 학습에 해당합니다.

---

### 4. Micro Batch 및 Dynamic Batching

| 항목 | FSDP | Megatron |
|---|---|---|
| `STUDENT_MICRO_BATCH_SIZE_PER_GPU` | 2 | 1 |
| `USE_DYNAMIC_BSZ` | `True` | `False` |
| `STUDENT_MAX_TOKEN_LEN_PER_GPU` | `2 × (256+512) = 1536` | `1 × (256+512) = 768` |

FSDP는 Dynamic Batch Size Packing을 활성화해 GPU 활용률을 높입니다. Megatron은 고정 배치 크기로 동작해 예측 가능한 메모리 사용을 유지합니다.

---

### 5. CPU Offload

| 항목 | FSDP | Megatron |
|---|---|---|
| `param_offload` | `True` | `True` |
| `optimizer_offload` | `True` | `True` |
| `grad_offload` | (없음) | `False` |

FSDP는 `fsdp_config` 네임스페이스, Megatron은 `megatron` 네임스페이스로 각각 설정됩니다.
Megatron에는 gradient offload 옵션이 추가로 존재하며 현재는 비활성화.

---

### 6. Activation Recompute (Gradient Checkpointing)

**FSDP 버전**: 단순 활성화:
```bash
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

**Megatron 버전**: 세밀한 제어 옵션 추가:
```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
```

`uniform` + `full` + `num_layers=1`은 모든 레이어에서 전체 activation을 균등하게 recompute하는 설정으로, 메모리를 최소화하는 대신 추가 연산이 발생합니다.

---

### 7. Megatron 전용 옵션 — mBridge

Megatron 버전에만 존재하는 가중치 전달 브릿지 설정:
```bash
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
actor_rollout_ref.actor.megatron.use_remove_padding=True
```

mBridge는 Megatron 텐서 병렬 레이아웃의 가중치를 vLLM rollout 엔진이 이해할 수 있는 형식으로 변환하는 인터페이스입니다.

---

### 8. CUDA Graph (ENFORCE_EAGER)

| 항목 | FSDP | Megatron |
|---|---|---|
| `ENFORCE_EAGER` | `True` | `False` |

FSDP는 eager 모드(CUDA graph 비활성화)로 디버깅에 유리하고, Megatron은 CUDA graph를 활성화하여 추론 성능을 극대화합니다.

---

## 종합 비교 요약

| 관점 | FSDP 버전 | Megatron 버전 |
|---|---|---|
| **주요 목적** | RL + 증류 혼합 학습 | 순수 증류 학습 |
| **GPU 요구량** | 최소 2 GPU | 최소 4 GPU |
| **학습 손실** | k1 + Policy Gradient | forward_kl_topk only |
| **병렬화** | FSDP 데이터 병렬 | 텐서 병렬(TP=2) |
| **배치 처리** | Dynamic packing | 고정 배치 |
| **추론 모드** | Eager (디버깅 친화) | CUDA graph (성능 우선) |
| **구현 복잡도** | 낮음 | 높음 |
| **메모리 제어** | FSDP 자동 관리 | Megatron 세밀 제어 |

---

## 참고 파일

- `verl/trainer/config/ppo_trainer.yaml` — FSDP 기본 config
- `verl/trainer/config/ppo_megatron_trainer.yaml` — Megatron 기본 config
- `verl/workers/fsdp_workers.py` — FSDP Worker 구현
- `verl/workers/megatron_workers.py` — Megatron Worker 구현
