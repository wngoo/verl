# verl On-Policy Distillation 리소스 풀 설정 가이드

---
date: 2026-04-08
topic: Student/Teacher world size, resource pool, Ulysses SP, gpu_memory_utilization 설정
tags: [verl, distillation, resource_pool, world_size, sequence_parallel, gpu_memory, multi-node]
---

## 개요

On-Policy Distillation 학습에서 Student/Teacher 모델의 GPU 배분을 제어하는 핵심 변수들.

---

## STUDENT_WORLD_SIZE

```bash
STUDENT_WORLD_SIZE=2  →  trainer.n_gpus_per_node=2
```

Student 모델(FSDP)이 사용하는 **노드당 GPU 수**. `global_pool` 크기를 결정한다.

```python
# main_ppo.py:228-229
resource_pool_spec = {
    "global_pool": [n_gpus_per_node] * nnodes
}
```

이 `global_pool`에 Student의 Actor/Rollout/Ref 워커가 모두 배치된다.

---

## TEACHER_RESOURCE_POOL

Teacher가 Student GPU를 공유하느냐(colocate), 전용 GPU를 갖느냐(standalone)를 결정한다.

### False — Colocate 모드 (기본값)

```python
# ray_trainer.py:513-514
def _should_compute_teacher_colocate(self):
    return use_teacher_policy and not enable_resource_pool  # True
```

- Teacher와 Student가 **동일 GPU를 시간 분할**로 사용
- Rollout → Teacher logprob 계산 → 학습 순서로 순차 실행
- GPU가 적을 때 사용
- `TEACHER_WORLD_SIZE`는 무시됨

### True — Standalone 모드

```python
# main_ppo.py:252-255
teacher_pool = [teacher_model.n_gpus_per_node] * teacher_model.nnodes
resource_pool_spec["teacher_pool"] = teacher_pool
```

- Teacher 전용 GPU pool 생성
- Student와 Teacher **동시 실행 가능** → 처리량 향상
- `TEACHER_WORLD_SIZE`가 실제로 적용됨

### 두 모드 비교

| | Colocate (False) | Standalone (True) |
|---|---|---|
| GPU 수 | Student GPU만 필요 | Student + Teacher GPU 필요 |
| 동작 | 순차 실행 | 동시 실행 |
| 처리량 | 낮음 | 높음 |
| 적합한 상황 | GPU 부족, 디버깅 | 대규모 학습 |

---

## TEACHER_WORLD_SIZE (Standalone 모드에서)

```python
# teacher_model.py:58-68
teacher_world_size = TP * DP * PP  # replica당 GPU 수
num_replicas = total_teacher_gpus // teacher_world_size
```

**실제 Teacher replica 수 = 총 Teacher GPU 수 / (TP × DP × PP)**

예시:
- Teacher GPU 16개, TP=1 → 16개 replica
- Teacher GPU 16개, TP=2 → 8개 replica

---

## SP (Ulysses Sequence Parallelism)

```bash
SP=1  →  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
```

시퀀스를 여러 GPU에 **토큰 단위로 분할**해 Attention 연산을 병렬화하는 기법(DeepSpeed Ulysses).

```python
# dp_actor.py:74-75
self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1  # SP=1이면 False
```

```
SP=1 (비활성화):  GPU0이 전체 시퀀스 처리
SP=2 (활성화):   GPU0: 앞 절반, GPU1: 뒤 절반 → AlltoAll 통신으로 합산
```

### max_token_len 자동 보정

```python
# dp_actor.py:563
max_token_len = ppo_max_token_len_per_gpu * ulysses_sequence_parallel_size
# SP=2: 1536 * 2 = 3072  (각 GPU가 절반만 처리하므로 예산 2배 필요)
```

### 언제 SP>1이 필요한가?

| 상황 | 권장 SP |
|------|---------|
| 짧은 시퀀스 (< 4K tokens) | 1 |
| 긴 시퀀스로 GPU 메모리 부족 | 2 이상 |
| SP는 반드시 WORLD_SIZE의 약수여야 함 | SP ≤ WORLD_SIZE |

---

## gpu_memory_utilization

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

**각 GPU VRAM 중 vLLM(rollout)의 KV cache에 예약하는 비율**이다. GPU 수가 아님.

```
각 Student GPU VRAM (예: 80GB):
├── vLLM KV Cache:    0.5 × 80GB = 40GB  (rollout용)
└── FSDP 학습:        나머지 ~40GB
```

Student GPU는 rollout과 학습을 **순차적으로** 번갈아 수행한다. vLLM이 시작 시 메모리를 미리 예약(pre-allocate)하기 때문에 FSDP를 위한 공간을 남겨두어야 한다.

### 값 설정 기준

| 상황 | 권장값 |
|------|--------|
| Colocate 모드 (Teacher도 같은 GPU) | 0.3 이하 |
| Standalone 모드 (Teacher 전용 GPU) | 0.5~0.7 |
| Student 모델이 클수록 | 낮게 설정 |

---

## 멀티노드 64GPU 권장 구성 (8 nodes × 8 GPUs)

```
Student (0.5B): 6 nodes × 8 GPUs = 48 GPUs  [FSDP, 데이터 병렬]
Teacher (3B):   2 nodes × 8 GPUs = 16 GPUs  [vLLM, TP=1, 16 replicas]
```

```bash
STUDENT_NNODES=6
STUDENT_WORLD_SIZE=8

TEACHER_RESOURCE_POOL=True
TEACHER_NNODES=2
TEACHER_WORLD_SIZE=8
TEACHER_TP=1
TEACHER_NUM_WORKERS=16

TRAIN_PROMPT_BSZ=1024  # 원본 128 → 스케일업

# GPU 여유로 offload 불필요
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# Teacher 전용 GPU 분리로 여유 생김
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### Ray 클러스터 실행

```bash
# Head node
ray start --head --port=6379 --num-gpus=8

# Worker nodes (각 노드에서)
ray start --address='<HEAD_IP>:6379' --num-gpus=8

# Head node에서 학습 실행
bash run_qwen_gsm8k_64gpu.sh
```

---

## 관련 파일/코드

- `verl/trainer/main_ppo.py:224` — `init_resource_pool_mgr`: resource pool 구성
- `verl/trainer/ppo/ray_trainer.py:513` — `_should_compute_teacher_colocate`
- `verl/experimental/teacher_loop/teacher_model.py:56` — Teacher replica 수 계산
- `verl/workers/config/distillation.py` — `DistillationTeacherModelConfig`
- `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` — 예제 설정
