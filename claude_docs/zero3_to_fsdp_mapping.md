---
date: 2026-04-09
topic: DeepSpeed ZeRO-3 설정을 verl FSDP 파라미터로 매핑하는 방법
tags: [fsdp, zero3, deepspeed, memory, offload, mixed_precision, distillation]
---

# DeepSpeed ZeRO-3 → verl FSDP 파라미터 매핑

## 개요

`trl_onpolicy_kd_gemma4_h100x64.md`의 DeepSpeed ZeRO-3 설정을 `run_qwen_gsm8k.sh` (verl FSDP 기반)에 적용하는 방법을 정리한다.

TRL은 DeepSpeed ZeRO-3 (`deepspeed_zero3.json`)를 쓰고, verl은 PyTorch FSDP를 쓰지만 개념적으로 1:1 매핑이 가능하다.

- 관련 스크립트: `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`
- 관련 문서: `claude_docs/trl_onpolicy_kd_gemma4_h100x64.md`
- FSDP 설정 클래스: `verl/workers/config/engine.py` → `FSDPEngineConfig`

---

## 핵심 매핑 테이블

| ZeRO-3 설정 (`deepspeed_zero3.json`) | verl FSDP 파라미터 | 비고 |
|---|---|---|
| `"stage": 3` | `reshard_after_forward=True` | **기본값** — 이미 ZeRO-3 동작 |
| `"overlap_comm": true` | `fsdp_config.forward_prefetch=True` | 기본 미설정 |
| `"bf16": {"enabled": true}` | `dtype=bfloat16` | 기본값 |
| `"communication_data_type": "bf16"` | `fsdp_config.mixed_precision.reduce_dtype=bf16` | 기본은 fp32 |
| `"offload_optimizer": {"device": "none"}` | `fsdp_config.optimizer_offload=False` | GPU 수에 따라 결정 |
| `"offload_param": {"device": "none"}` | `fsdp_config.param_offload=False` | GPU 수에 따라 결정 |
| `"activation_checkpointing": {...}` | `model.enable_gradient_checkpointing=True` | 이미 설정됨 |
| `"gradient_clipping": 1.0` | `actor.optim.clip_grad=1.0` | 기본 미설정 |

---

## 상세 설명

### 1. `stage: 3` → `reshard_after_forward=True`

ZeRO-3의 핵심은 파라미터까지 GPU간 분산하는 것.

```
ZeRO-2: 옵티마이저 상태 + 그래디언트 분산 (파라미터는 각 GPU에 full copy)
ZeRO-3: 옵티마이저 상태 + 그래디언트 + 파라미터 분산

FSDP reshard_after_forward=True  → ZeRO-3 (forward 후 파라미터 분산 = FULL_SHARD)
FSDP reshard_after_forward=False → ZeRO-2 (파라미터 유지 = SHARD_GRAD_OP)
```

`run_qwen_gsm8k.sh`는 기본값(`True`)을 사용하므로 이미 ZeRO-3 동작이다.

---

### 2. `overlap_comm: true` → `forward_prefetch=True`

```
ZeRO-3:  overlap_comm=true    → 통신(allgather)과 연산을 중첩 → 속도 향상
FSDP:    forward_prefetch=True → 다음 레이어의 파라미터를 미리 allgather (같은 효과)
```

`fsdp_workers.py`에서 `FSDPEngineConfig.forward_prefetch`로 제어된다.

---

### 3. `communication_data_type: "bf16"` → `mixed_precision.reduce_dtype=bf16`

H100에서 gradient AllReduce를 FP32 대신 BF16으로 수행하여 대역폭 절약.

`fsdp_workers.py:571-581` 참고:
```python
mixed_precision_config = fsdp_config.get("mixed_precision", None)
if mixed_precision_config is not None:
    param_dtype  = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype",  "bf16"))
    reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
    buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
```

`mixed_precision`을 명시하지 않으면 `reduce_dtype`이 FP32로 고정된다.

---

### 4. Offload 설정 — GPU 수에 따라 결정

ZeRO-3 doc 핵심 조언: GPU가 충분하면 offload 불필요 (ZeRO-3만으로 메모리 분산 효과 충분).

**메모리 추정 (ZeRO-3 기준):**
```
단일 GPU 총 메모리 = 파라미터(BF16) + 그래디언트(BF16) + 옵티마이저(FP32)
0.5B 모델: 1GB + 1GB + 4GB = 6GB 전체
ZeRO-3 N GPU: 6GB / N per GPU (steady state)

N=2:  3.0 GB/GPU  (+ 활성화 메모리)
N=8:  0.75 GB/GPU (여유 충분 → offload 불필요)
N=32: 0.19 GB/GPU
```

**GPU 수별 권장 offload 설정:**

| GPU 수 | `param_offload` | `optimizer_offload` |
|--------|----------------|---------------------|
| 2 (현재 기본) | `True` (안전) | `True` (안전) |
| 8 이상 | `False` | `False` |
| 32 (ZeRO-3 doc 기준) | `False` | `False` |

---

## 코드 예시: 8 GPU로 확장 시 STUDENT 설정

```bash
STUDENT_WORLD_SIZE=8   # 2 → 8로 확장

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.clip_grad=1.0                   # gradient_clipping 1.0

    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ

    # ZeRO-3 offload_optimizer/param: "none" 에 해당
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    # ZeRO-3 overlap_comm: true 에 해당
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True

    # ZeRO-3 communication_data_type: bf16 에 해당
    # 주의: 중첩 dict이므로 Hydra에서 따옴표로 감싸야 함
    'actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16'
    'actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16'
    'actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32'

    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)
```

---

## 코드 예시: 2 GPU 유지 시 적용 가능한 설정만

GPU를 늘리지 않는 상황이라면 offload는 유지하고 통신/속도 설정만 추가:

```bash
STUDENT=(
    # 기존 offload 유지
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

    # 추가 가능한 ZeRO-3 설정
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    'actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16'
)
```

---

## 관련 파일/코드

- `verl/workers/config/engine.py:211` — `FSDPEngineConfig` 클래스 정의
  - `reshard_after_forward`: bool, 기본 `True`
  - `forward_prefetch`: bool, 기본 `False`
  - `param_offload`: bool, 기본 `False`
  - `optimizer_offload`: bool, 기본 `False`
  - `mixed_precision`: Optional[dict], 기본 `None` (→ reduce_dtype=fp32)
- `verl/workers/fsdp_workers.py:114` — `get_sharding_strategy()`: `reshard_after_forward` → `FULL_SHARD` 또는 `SHARD_GRAD_OP`
- `verl/workers/fsdp_workers.py:571` — `mixed_precision` 설정 파싱 로직
- `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` — 적용 대상 스크립트

---

## 참고 사항

- `mixed_precision` 설정은 Hydra에서 nested dict이므로 반드시 `'...'` 따옴표로 감싸야 parse 오류가 없다.
- verl actor의 `cpu_offload`는 코드상 `role == "actor"`일 때 `None`으로 강제 설정된다 (`fsdp_workers.py:607`). `param_offload=True`는 별도 manual offload 로직이고 FSDP 내부 CPUOffload와 다른 경로다.
- `stage3_param_persistence_threshold`(ZeRO-3에서 작은 파라미터는 모든 GPU에 유지)에 직접 대응하는 FSDP 파라미터는 없다. FSDP의 `wrap_policy.min_num_params`로 wrap 단위 크기 조절이 가장 유사한 접근이다.
- ZeRO-3 doc에서 `sub_group_size`, `stage3_max_live_parameters`, `stage3_max_reuse_distance`는 FSDP에 직접 대응 없음 (FSDP는 module 단위로 자동 관리).
