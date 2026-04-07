# On-Policy Knowledge Distillation (On-Policy KD)

> 분석 기준 스크립트: `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`

---

## 목차

1. [개요](#1-개요)
2. [핵심 설정값](#2-핵심-설정값)
3. [Teacher 모델의 Inference 엔진](#3-teacher-모델의-inference-엔진)
4. [GPU 배치 구조](#4-gpu-배치-구조)
5. [초기화 흐름](#5-초기화-흐름)
6. [학습 루프 상세](#6-학습-루프-상세)
7. [Loss 함수 상세](#7-loss-함수-상세)
8. [Wake-up / Sleep 메커니즘](#8-wake-up--sleep-메커니즘)
9. [Teacher 배치 모드 비교](#9-teacher-배치-모드-비교)
10. [코드 위치 참조](#10-코드-위치-참조)

---

## 1. 개요

On-Policy KD는 **student 모델이 자체 생성한 응답**에 대해 teacher 모델의 token-level log probability를 supervision signal로 사용하는 학습 방식이다.

- 기존 offline KD: teacher가 생성한 텍스트로 student를 학습
- On-Policy KD: **student가 생성한 텍스트**에 대한 teacher의 log prob으로 학습

이 방식은 student 분포에 맞춰 teacher signal을 얻으므로 distribution shift 문제를 줄이고, PPO-style policy gradient와 결합하여 안정적인 학습이 가능하다.

---

## 2. 핵심 설정값

```bash
# 모델
STUDENT_MODEL=Qwen2.5-0.5B
TEACHER_MODEL=Qwen2.5-3B-Instruct

# Rollout/Inference 엔진 (student와 teacher 모두 동일하게 적용)
ROLLOUT_NAME="vllm"        # "sglang" 으로 변경 가능

# Teacher 배치 방식
TEACHER_RESOURCE_POOL=False  # False=standalone, True=colocated
TEACHER_WORLD_SIZE=4          # teacher 전용 GPU 수

# Student 학습
STUDENT_WORLD_SIZE=2          # student 전용 GPU 수

# Distillation Loss
DISTILLATION_LOSS_MODE="k1"           # reverse KL 단일 샘플 추정
USE_POLICY_GRADIENT=True              # KL을 reward로 변환해 PPO 적용
DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0
```

---

## 3. Teacher 모델의 Inference 엔진

**Teacher도 vLLM (또는 SGLang)**을 사용한다.

`distillation.yaml`의 기본값:

```yaml
# verl/trainer/config/distillation/distillation.yaml
teacher_model:
  inference:
    name: ${oc.select:actor_rollout_ref.rollout.name}  # student rollout과 동일한 엔진
```

스크립트에서 `distillation.teacher_model.inference.name=$ROLLOUT_NAME`으로 명시적으로 오버라이드하므로 `ROLLOUT_NAME` 하나로 student rollout과 teacher inference 모두 제어된다.

### 엔진 선택 흐름

```
ROLLOUT_NAME="vllm"
      │
      ├─ actor_rollout_ref.rollout.name=vllm    → Student rollout에 vLLMReplica 사용
      └─ distillation.teacher_model.inference.name=vllm → Teacher에 vLLMReplica 사용
```

내부적으로 `get_rollout_replica_class(name)` (`verl/workers/rollout/replica.py:396`) 가 엔진 이름을 보고 `vLLMReplica` 또는 `SGLangReplica`를 반환한다.

---

## 4. GPU 배치 구조

스크립트 기준 (`STUDENT_WORLD_SIZE=2`, `TEACHER_WORLD_SIZE=4`, `TEACHER_RESOURCE_POOL=False`):

```
┌─────────────────────────────────────────────────────┐
│                  Ray Cluster                        │
│                                                     │
│  ┌──────────────────────────────┐                   │
│  │  Student Resource Pool       │                   │
│  │  GPU 0, 1                    │                   │
│  │  - FSDP Actor (학습)         │                   │
│  │  - vLLM hybrid (rollout)     │                   │
│  └──────────────────────────────┘                   │
│                                                     │
│  ┌──────────────────────────────┐                   │
│  │  Teacher Resource Pool       │  (standalone 모드) │
│  │  GPU 2, 3, 4, 5              │                   │
│  │  - vLLM Server (logprob 전용)│                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

> `TEACHER_RESOURCE_POOL=True` (colocated 모드)로 바꾸면 teacher가 별도 pool 없이 student와 같은 placement group 내에서 별도 프로세스로 실행된다.

---

## 5. 초기화 흐름

```
python -m verl.trainer.main_ppo
  └─ RayPPOTrainer.__init__()        (verl/trainer/ppo/ray_trainer.py)
       │
       ├─ [1] ActorRolloutRefWorker 생성 (FSDP + vLLM hybrid)
       │       → GPU 0, 1 점유
       │       → distillation_config도 함께 전달 (topk logits 계산용)
       │
       └─ [2] TeacherModelManager.__init__()   (verl/experimental/teacher_loop/teacher_model.py)
               │
               ├─ _initialize_llm_servers()
               │     get_rollout_replica_class("vllm")  → vLLMReplica
               │     num_replicas = 4 // 1 = 4  (world_size=4, tp=1이면 4개 replica)
               │     vLLMReplica.init_standalone()
               │       → 새 resource pool 생성 (4 GPU)
               │       → Ray worker group → vLLM HTTP 서버 기동
               │
               ├─ _initialize_async_server_manager()
               │     GlobalRequestLoadBalancer (Ray actor) 생성
               │     AsyncTeacherLLMServerManager 생성 (HTTP 클라이언트 래퍼)
               │
               ├─ _initialize_router()
               │     naive_router 프로세스 기동 (load balancing)
               │
               └─ sleep()  ← 초기화 후 즉시 절전 상태로 대기
```

---

## 6. 학습 루프 상세

한 step에서 일어나는 일을 순서대로 정리한다.

### Step 1 — Student Rollout

```
ActorRolloutRefWorker (GPU 0, 1)
  - student 모델(Qwen2.5-0.5B)로 각 prompt에 대해 response 생성
  - 출력: DataProto { input_ids, responses, attention_mask, position_ids, ... }
  - n=1 (샘플 1개), max_response_length=512
```

### Step 2 — Teacher Log Probability 계산

```
TeacherModelManager.compute_logprobs(batch)    (verl/experimental/teacher_loop/teacher_model.py:127)
  │
  ├─ wake_up()   ← teacher vLLM 서버 활성화 (GPU 2~5)
  │
  ├─ AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(batch)
  │     (verl/experimental/teacher_loop/teacher_manager.py:128)
  │     │
  │     ├─ 배치 내 각 샘플을 asyncio.gather로 병렬 처리
  │     │
  │     └─ 각 샘플:
  │           _unpad_teacher_inputs()
  │             → left-padding 제거, 유효한 prompt+response 시퀀스 추출
  │           vLLM HTTP API 호출:
  │             max_tokens=1          ← 실제 텍스트 생성 없음
  │             prompt_logprobs=0     ← k1 모드: 단일 log prob만 필요 (topk=0)
  │             (forward_kl_topk 모드라면 prompt_logprobs=64)
  │           응답에서 teacher_ids, teacher_logprobs 추출
  │           _pad_teacher_outputs()  → 원래 배치 shape으로 패딩
  │
  ├─ 반환: DataProto { teacher_ids: (B, S), teacher_logprobs: (B, S, 1 or K) }
  │
  └─ sleep()   ← GPU 메모리를 student 학습에 양보
```

> **핵심**: teacher는 텍스트를 생성하는 것이 아니라, student가 이미 생성한 시퀀스에 대해 `prompt_logprobs`만 계산한다. `max_tokens=1`로 실제 새 토큰 생성을 최소화한다.

### Step 3 — Advantage 계산

```
- teacher_logprobs를 batch에 병합
- algorithm.adv_estimator=grpo 방식으로 advantage 계산
  (task reward가 없으므로 distillation loss 기반 reward가 주된 학습 신호)
```

### Step 4 — Student 학습 (FSDP)

```
ActorRolloutRefWorker (GPU 0, 1) — 학습 단계
  │
  ├─ distillation_loss 계산  (verl/trainer/distillation/losses.py)
  │     k1 모드:
  │       k1 = exp(log q_student - log p_teacher) - 1
  │
  ├─ use_policy_gradient=True:
  │     advantages = -distillation_losses.detach()   ← KL을 음수로 뒤집어 reward로 사용
  │     policy_loss_fn(old_log_prob, log_prob, advantages, ...)   ← PPO clip 적용
  │
  └─ optimizer.step()
```

---

## 7. Loss 함수 상세

### 지원 Loss 모드

| 모드 | 방식 | 설명 |
|---|---|---|
| `k1` | estimator | `exp(log_q - log_p) - 1` — reverse KL 단일 샘플 추정, 음수 가능 |
| `kl` | estimator | `log_q - log_p` — 표준 KL |
| `k3` | estimator | `0.5 * (log_q - log_p)^2` — 분산이 낮은 추정 |
| `forward_kl_topk` | topk | teacher top-K logit으로 forward KL 계산 |
| `abs`, `mse`, `k2`, `low_var_kl` | estimator | 기타 변형 |

### k1 + use_policy_gradient=True (스크립트 기본값)

```python
# verl/trainer/distillation/losses.py:253-269

# 1. token-level distillation loss 계산
k1 = exp(log_q_student - log_p_teacher) - 1   # (B, T)

# 2. 음수로 뒤집어 advantage로 변환 (teacher와 가까울수록 reward ↑)
advantages = -k1.detach()

# 3. PPO policy gradient 적용
policy_loss = ppo_clip(old_log_prob, log_prob, advantages)
```

`use_task_rewards=False`이므로 PPO advantage에 task reward(정답 여부)는 포함되지 않고, 순수하게 teacher 분포 모방을 목표로 한다.

### use_policy_gradient=False인 경우 (직접 역전파)

```python
# 직접 distillation loss를 supervised loss로 역전파
loss = agg_loss(distillation_losses, response_mask)
```

---

## 8. Wake-up / Sleep 메커니즘

Teacher와 student가 같은 GPU를 시간 다중화로 공유할 수 있도록, teacher 서버는 필요할 때만 활성화된다.

```
학습 타임라인:

  [Student Rollout] ──────────────────┐
                                      ▼
                              [Teacher wake_up]
                              [Teacher logprob 계산]
                              [Teacher sleep]
                                      │
                                      ▼
                              [Advantage 계산]
                                      │
                                      ▼
                              [Student 학습 (FSDP)]
                                      │
                    ┌─────────────────┘
                    ▼
             다음 iteration...
```

이 설계 덕분에 `TEACHER_RESOURCE_POOL=False` (standalone)이어도 총 6개 GPU(2+4) 대신 더 적은 GPU로도 운용이 가능하다 — teacher 절전 중에는 student 학습이 해당 GPU를 활용할 수 있다.

---

## 9. Teacher 배치 모드 비교

| 설정 | `TEACHER_RESOURCE_POOL=False` (standalone) | `TEACHER_RESOURCE_POOL=True` (colocated) |
|---|---|---|
| GPU pool | teacher 전용 별도 pool 생성 | student와 같은 placement group 공유 |
| 프로세스 | 완전히 독립적 | 같은 placement group, 별도 프로세스 |
| 유연성 | teacher/student GPU 수 독립 지정 | GPU 공유로 메모리 효율적 |
| 코드 경로 | `init_standalone()` | `init_colocated()` |

---

## 10. 코드 위치 참조

| 컴포넌트 | 파일 | 설명 |
|---|---|---|
| 학습 진입점 | `verl/trainer/main_ppo.py` | Hydra 기반 학습 실행 |
| 학습 루프 | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer`, teacher logprob 계산 호출 |
| Teacher 관리자 | `verl/experimental/teacher_loop/teacher_model.py` | `TeacherModelManager` |
| Teacher async client | `verl/experimental/teacher_loop/teacher_manager.py` | `AsyncTeacherLLMServerManager` |
| Rollout replica | `verl/workers/rollout/replica.py` | `vLLMReplica`, `SGLangReplica` 등록 및 팩토리 |
| Distillation loss | `verl/trainer/distillation/losses.py` | k1, forward_kl_topk 등 loss 구현 |
| Distillation 설정 | `verl/trainer/config/distillation/distillation.yaml` | 기본 config 값 |
| Worker config | `verl/workers/config.py` | `DistillationConfig`, `DistillationTeacherModelConfig` |
