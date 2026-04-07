# GPU 세팅 파라미터 완전 해설

**기반 스크립트:** `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`  
**목적:** 각 파라미터의 역할, 상호 관계, 멀티 노드 적용 시 조정 방법을 설명

---

## 목차

1. [파라미터 전체 관계도](#1-파라미터-전체-관계도)
2. [데이터 관련 파라미터](#2-데이터-관련-파라미터)
3. [Student Actor 학습 파라미터](#3-student-actor-학습-파라미터)
4. [FSDP Offload 파라미터](#4-fsdp-offload-파라미터)
5. [Student Rollout 파라미터](#5-student-rollout-파라미터)
6. [Teacher 추론 파라미터](#6-teacher-추론-파라미터)
7. [멀티 노드 조정 가이드](#7-멀티-노드-조정-가이드)
8. [요약 치트시트](#8-요약-치트시트)

---

## 1. 파라미터 전체 관계도

```
TRAIN_PROMPT_BSZ  (글로벌 배치: 1 step당 총 시퀀스 수)
    │
    └── ppo_mini_batch_size  (1회 gradient update에 사용할 시퀀스 수)
            │
            ├── [use_dynamic_bsz=False]
            │     ppo_micro_batch_size_per_gpu × DP_size × gradient_accumulation = ppo_mini_batch_size
            │
            └── [use_dynamic_bsz=True]
                  ppo_max_token_len_per_gpu 기준으로 시퀀스를 동적 packing → micro-batch 구성

DP (Data Parallelism) = trainer.n_gpus_per_node × trainer.nnodes / SP
```

---

## 2. 데이터 관련 파라미터

```bash
MAX_PROMPT=256              # 프롬프트 최대 토큰 수
MAX_RESPONSE_LENGTH=512     # 응답 최대 토큰 수
MAX_NUM_TOKENS=769          # MAX_PROMPT + MAX_RESPONSE_LENGTH + 1
TRAIN_PROMPT_BSZ=128        # 글로벌 배치 크기
```

### `MAX_PROMPT` / `MAX_RESPONSE_LENGTH`

- rollout, log prob 계산, teacher 추론 모두에 적용되는 **시퀀스 길이 상한**
- 값이 크면 더 긴 생성이 가능하지만 KV cache 메모리 증가

### `TRAIN_PROMPT_BSZ` (= `data.train_batch_size`)

- 1 training step에서 rollout → teacher → actor update까지 처리하는 **총 시퀀스 수**
- **멀티 노드에서 DP가 N배 늘면 N배 증가 권장**
  - GPU당 처리 시퀀스 수 = `TRAIN_PROMPT_BSZ / DP`를 일정하게 유지하기 위함
  - 단, on-policy KD에서는 Teacher 처리 부담이 커지므로 과도하게 키우지 않을 것

---

## 3. Student Actor 학습 파라미터

```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=1536   # = 2 × (256 + 512)
USE_DYNAMIC_BSZ=True
SP=1
```

### `ppo_mini_batch_size`

- 글로벌 배치에서 **1회 gradient update**에 사용하는 시퀀스 수
- 스크립트에서 `TRAIN_PROMPT_BSZ`와 동일하게 설정 = on-policy 방식 (step당 1회 업데이트)
- `ppo_mini_batch_size < TRAIN_PROMPT_BSZ`로 설정하면 복수의 mini-batch로 나뉘어 여러 번 업데이트 (PPO 스타일)

```
mini_batches 수 = TRAIN_PROMPT_BSZ / ppo_mini_batch_size
               = 128 / 128 = 1  (on-policy, 1회 업데이트)
```

### `ppo_micro_batch_size_per_gpu` (= `STUDENT_MICRO_BATCH_SIZE_PER_GPU`)

- GPU 1개가 **한 번의 forward/backward pass에 처리하는 시퀀스 수**
- `use_dynamic_bsz=False`일 때만 직접 사용됨
- **GPU당 메모리 기준이므로 멀티 노드에서 변경 불필요**

```
[use_dynamic_bsz=False일 때]
gradient_accumulation = ppo_mini_batch_size / (ppo_micro_batch_size_per_gpu × DP)
                      = 128 / (2 × 2) = 32  (1노드 2GPU 기준)
```

### `use_dynamic_bsz` (= `USE_DYNAMIC_BSZ`)

| 값 | 동작 | 특징 |
|---|---|---|
| `True` | 토큰 수 기준으로 micro-batch를 동적 구성 | GPU 활용률 최대화, 가변 길이 시퀀스에 유리 |
| `False` | 고정 시퀀스 수로 micro-batch 구성 | 예측 가능한 메모리 사용, padding 낭비 발생 |

동작 원리 (`dp_actor.py:562-566`):
```python
if use_dynamic_bsz:
    max_token_len = ppo_max_token_len_per_gpu × ulysses_sp_size
    micro_batches = prepare_dynamic_batch(data, max_token_len=max_token_len)
else:
    micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)
```

### `ppo_max_token_len_per_gpu` (= `STUDENT_MAX_TOKEN_LEN_PER_GPU`)

- `use_dynamic_bsz=True`일 때 **micro-batch 1개의 최대 토큰 수 (GPU 1개 기준)**
- GPU 메모리를 직접 제어하는 핵심 값
- 계산 공식:

```bash
STUDENT_MAX_TOKEN_LEN_PER_GPU = STUDENT_MICRO_BATCH_SIZE_PER_GPU × (MAX_PROMPT + MAX_RESPONSE_LENGTH)
                               = 2 × (256 + 512) = 1536
```

- 값이 크면 GPU 메모리 사용 증가, 처리량 향상
- 값이 작으면 메모리 안전, GPU 활용률 저하
- **멀티 노드에서 GPU당 메모리는 동일하므로 변경 불필요**

### `ulysses_sequence_parallel_size` (= `SP`)

- Ulysses Sequence Parallelism 크기
- `SP=1`: 비활성화 (일반 Data Parallelism)
- `SP>1`: 1개 시퀀스를 여러 GPU에 분할 처리 → 매우 긴 시퀀스(8k+)에 유효

```
DP = (trainer.n_gpus_per_node × trainer.nnodes) / SP
SP=1, 8GPU, 1노드 → DP = 8
SP=2, 8GPU, 1노드 → DP = 4  (단, 각 시퀀스를 2 GPU가 처리)
```

> **주의**: `SP > 1`이면 `use_remove_padding=True`가 반드시 필요 (`actor.py:321-325`)

---

## 4. FSDP Offload 파라미터

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

| 파라미터 | 역할 | GPU VRAM 절감 | 속도 영향 |
|---|---|---|---|
| `param_offload=True` | 모델 파라미터를 CPU RAM에 보관, 연산 시만 GPU로 이동 | 크게 절감 | 약간 느림 |
| `optimizer_offload=True` | Adam 모멘텀 등 optimizer state를 CPU에 보관 | 절감 | 약간 느림 |

- Qwen2.5-0.5B처럼 소형 모델은 offload를 켜도 속도 영향이 크지 않음
- 멀티 노드에서도 동일 설정 유지 권장

---

## 5. Student Rollout 파라미터

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
actor_rollout_ref.rollout.max_model_len=769
actor_rollout_ref.rollout.max_num_batched_tokens=769
actor_rollout_ref.rollout.max_num_seqs=769
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1536
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
```

### `tensor_model_parallel_size` (Student Rollout TP)

- Student vLLM 롤아웃 엔진의 Tensor Parallelism
- `TP=1`: Qwen2.5-0.5B은 1 GPU로 충분
- `TP=N`이면 N개 GPU가 하나의 vLLM 인스턴스를 공유

### `gpu_memory_utilization`

- vLLM KV cache에 할당하는 **GPU 메모리 비율**
- `0.3`으로 낮게 설정한 이유: FSDP 학습과 같은 GPU를 공유(colocated)하기 때문
- `enable_resource_pool=True`로 전용 GPU 분리 시 `0.6~0.8`로 상향 가능

### `max_model_len` / `max_num_batched_tokens` / `max_num_seqs`

| 파라미터 | 역할 |
|---|---|
| `max_model_len` | vLLM이 처리할 수 있는 최대 sequence 길이 |
| `max_num_batched_tokens` | 1 iteration에 처리할 최대 총 토큰 수 |
| `max_num_seqs` | 동시에 처리할 최대 sequence 수 |

세 값을 `MAX_NUM_TOKENS=769`로 동일하게 설정하면 한 번에 시퀀스 1개씩 처리 (안정성 우선).  
처리량을 높이려면 `max_num_seqs`와 `max_num_batched_tokens`를 키울 수 있음.

### `log_prob_micro_batch_size_per_gpu` / `log_prob_max_token_len_per_gpu`

- Teacher가 생성한 시퀀스에 대해 **Student가 log prob을 재계산**할 때의 배치 설정
- Actor update의 `ppo_micro_batch_size_per_gpu`, `ppo_max_token_len_per_gpu`와 동일한 값 사용 권장

---

## 6. Teacher 추론 파라미터

```bash
distillation.num_workers=8
distillation.teacher_model.n_gpus_per_node=4
distillation.teacher_model.nnodes=1
distillation.teacher_model.inference.tensor_model_parallel_size=1
distillation.teacher_model.inference.gpu_memory_utilization=0.3
distillation.teacher_model.inference.max_model_len=769
distillation.teacher_model.inference.max_num_batched_tokens=769
distillation.teacher_model.inference.max_num_seqs=769
```

### `distillation.num_workers`

- Teacher vLLM **replica 수 (병렬 추론 인스턴스 수)**
- 계산 공식:

```
num_workers = (teacher_n_gpus_per_node × teacher_nnodes) / teacher_TP
```

- 높을수록 Teacher 처리량 증가 → Student 학습 대기 시간 감소

### `tensor_model_parallel_size` (Teacher TP)

- Teacher vLLM의 Tensor Parallelism
- Qwen2.5-3B: GPU 1개(TP=1)에도 올라가지만, TP를 높이면 추론 속도 향상

### `gpu_memory_utilization` (Teacher)

- Teacher vLLM의 KV cache GPU 메모리 비율
- `enable_resource_pool=True`로 전용 노드 분리 시 `0.6~0.7`로 상향 권장

---

## 7. 멀티 노드 조정 가이드

### 핵심 원칙

| 파라미터 | 멀티 노드 시 조정 | 이유 |
|---|---|---|
| `TRAIN_PROMPT_BSZ` | DP 비율로 증가 | GPU당 시퀀스 수를 일정하게 유지 |
| `ppo_mini_batch_size` | `TRAIN_PROMPT_BSZ`와 동일 | on-policy 1회 업데이트 유지 |
| `ppo_micro_batch_size_per_gpu` | **변경 불필요** | GPU당 메모리 기준 |
| `ppo_max_token_len_per_gpu` | **변경 불필요** | GPU당 메모리 기준 |
| `gpu_memory_utilization` | 전용 GPU면 0.6~0.7로 상향 | resource pool 분리 효과 |
| `distillation.num_workers` | `teacher_total_gpus / teacher_TP` 공식 | 총 Teacher GPU 수 변화 반영 |
| `trainer.nnodes` | Student 노드 수 | 직접 지정 |
| `distillation.teacher_model.nnodes` | Teacher 노드 수 | `enable_resource_pool=True` 필수 |

### 노드 구성별 설정값 비교표

| 환경 | DP | `TRAIN_PROMPT_BSZ` | `ppo_mini_batch_size` | `ppo_micro_batch_size_per_gpu` | `ppo_max_token_len_per_gpu` |
|---|---|---|---|---|---|
| 1노드 × 2GPU (원본) | 2 | 128 | 128 | 2 | 1536 |
| 1노드 × 8GPU | 8 | 512 | 512 | 2 | 1536 |
| 2노드 × 8GPU (Student) | 16 | 512 | 512 | 2 | 1536 |
| 4노드 × 8GPU (Student) | 32 | 1024 | 1024 | 2 | 1536 |

> `ppo_micro_batch_size_per_gpu`와 `ppo_max_token_len_per_gpu`는 GPU 단위 메모리 기준이므로 노드 수에 무관하게 고정.

### `num_workers` (Teacher replica 수) 계산 예시

| Teacher 구성 | teacher_TP | num_workers |
|---|---|---|
| 1노드 × 4GPU | 1 | 4 |
| 1노드 × 4GPU | 2 | 2 |
| 1노드 × 8GPU | 4 | 2 |
| 2노드 × 8GPU | 4 | 4 |

### 2노드 × 8GPU 완전 설정 예시

```bash
# ── 기본 설정 ─────────────────────────────────────────────
MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))  # 769

# ── 멀티 노드 핵심 값 ──────────────────────────────────────
TRAIN_PROMPT_BSZ=512               # 원본(128) × DP증가배율(16/2=8) → 적절히 조정
                                   # 단, Teacher 부담 고려 시 256~512 범위 권장

STUDENT_NNODES=1                   # Student 전용 노드 수
STUDENT_WORLD_SIZE=8               # 노드당 Student GPU 수
# → DP = 8 × 1 = 8

STUDENT_MICRO_BATCH_SIZE_PER_GPU=2 # 변경 불필요
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
# = 1536  (변경 불필요)
USE_DYNAMIC_BSZ=True
SP=1

TEACHER_RESOURCE_POOL=True         # 전용 노드 분리 필수
TEACHER_NNODES=1                   # Teacher 전용 노드 수
TEACHER_WORLD_SIZE=8               # 노드당 Teacher GPU 수
TEACHER_TP=4                       # 3B 모델에 적합
TEACHER_NUM_WORKERS=2              # = 8 / 4

# ── Student 파라미터 ────────────────────────────────────────
STUDENT=(
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ          # 512
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU  # 2
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU        # 1536
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP             # 1
)

# ── Student Rollout 파라미터 ────────────────────────────────
ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 전용 GPU이므로 상향
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
)

# ── Teacher 파라미터 ────────────────────────────────────────
DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=$TEACHER_NUM_WORKERS                          # 2
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL # True
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE         # 8
    distillation.teacher_model.nnodes=$TEACHER_NNODES                      # 1
    distillation.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP  # 4
    distillation.teacher_model.inference.gpu_memory_utilization=0.7        # 전용 노드이므로 높게
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_TOKENS
    ...
)

# ── Trainer 파라미터 ────────────────────────────────────────
TRAINER=(
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE   # 8
    trainer.nnodes=$STUDENT_NNODES                # 1
    ray_kwargs.ray_init.num_cpus=64               # Slurm 환경 필수
    ...
)
```

---

## 8. 요약 치트시트

| 파라미터 | 스크립트 변수 | 기본값 | 역할 | 멀티 노드 조정 |
|---|---|---|---|---|
| `data.train_batch_size` | `TRAIN_PROMPT_BSZ` | 128 | 1 step 총 시퀀스 수 | **DP 비율 증가** |
| `actor.ppo_mini_batch_size` | `TRAIN_PROMPT_BSZ` | 128 | 1회 update 시퀀스 수 | `TRAIN_PROMPT_BSZ`와 동일 |
| `actor.ppo_micro_batch_size_per_gpu` | `STUDENT_MICRO_BATCH_SIZE_PER_GPU` | 2 | GPU당 forward/backward 시퀀스 수 | **불변** |
| `actor.ppo_max_token_len_per_gpu` | `STUDENT_MAX_TOKEN_LEN_PER_GPU` | 1536 | GPU당 최대 토큰 수 (dynamic bsz) | **불변** |
| `actor.use_dynamic_bsz` | `USE_DYNAMIC_BSZ` | True | 동적 배치 packing 활성화 | True 유지 권장 |
| `actor.ulysses_sequence_parallel_size` | `SP` | 1 | Sequence Parallelism | 긴 시퀀스에서만 > 1 |
| `actor.fsdp_config.param_offload` | — | True | 파라미터 CPU offload | 불변 |
| `actor.fsdp_config.optimizer_offload` | — | True | optimizer CPU offload | 불변 |
| `rollout.tensor_model_parallel_size` | — | 1 | Student vLLM TP | 0.5B는 1 유지 |
| `rollout.gpu_memory_utilization` | — | 0.3 | Student KV cache 비율 | 전용 GPU면 0.6~0.7 |
| `rollout.log_prob_micro_batch_size_per_gpu` | `STUDENT_MICRO_BATCH_SIZE_PER_GPU` | 2 | log prob 재계산 배치 수 | **불변** |
| `rollout.log_prob_max_token_len_per_gpu` | `STUDENT_MAX_TOKEN_LEN_PER_GPU` | 1536 | log prob 재계산 최대 토큰 | **불변** |
| `distillation.num_workers` | `TEACHER_NUM_WORKERS` | 8 | Teacher replica 수 | `teacher_GPU수 / teacher_TP` |
| `distillation.teacher_model.n_gpus_per_node` | `TEACHER_WORLD_SIZE` | 4 | 노드당 Teacher GPU 수 | 노드별 GPU 수로 설정 |
| `distillation.teacher_model.nnodes` | `TEACHER_NNODES` | (없음) | Teacher 노드 수 | 직접 지정 |
| `distillation.teacher_model.inference.tensor_model_parallel_size` | `TEACHER_TP` | 1 | Teacher vLLM TP | GPU/모델 크기에 따라 조정 |
| `distillation.teacher_model.inference.gpu_memory_utilization` | — | 0.3 | Teacher KV cache 비율 | 전용 노드면 0.6~0.7 |
| `trainer.n_gpus_per_node` | `STUDENT_WORLD_SIZE` | 2 | 노드당 Student GPU 수 | 실제 GPU 수 입력 |
| `trainer.nnodes` | `STUDENT_NNODES` | 1 | Student 노드 수 | 직접 지정 |

---

## 참고 코드 위치

| 파라미터 | 정의 위치 | 사용 위치 |
|---|---|---|
| `ppo_mini_batch_size` / `ppo_micro_batch_size_per_gpu` | `verl/workers/config/actor.py:143-148` | `verl/workers/actor/dp_actor.py:552-571` |
| `use_dynamic_bsz` / `ppo_max_token_len_per_gpu` | `verl/workers/config/actor.py:147-148` | `verl/workers/actor/dp_actor.py:562-566` |
| `gpu_memory_utilization` / `tensor_model_parallel_size` | `verl/workers/config/rollout.py:196-203` | vLLM 엔진 초기화 시 |
| `distillation.num_workers` / `teacher_model` 설정 | `verl/workers/config/distillation.py:114-136` | `verl/trainer/ppo/ray_trainer.py` |
| resource pool 생성 로직 | `verl/trainer/main_ppo.py:224-263` | Ray 클러스터 초기화 시 |
