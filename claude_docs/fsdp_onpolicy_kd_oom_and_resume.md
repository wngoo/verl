# FSDP On-Policy KD: CUDA OOM 해결 및 학습 Resume 가이드

> 분석 기준 스크립트: `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`
> 에러 위치: `Variable._execution_engine.run_backward`

---

## 목차

1. [에러 발생 위치와 원인](#1-에러-발생-위치와-원인)
2. [FSDP 학습 코드 흐름](#2-fsdp-학습-코드-흐름)
3. [micro_batch에 들어있는 값](#3-micro_batch에-들어있는-값)
4. [TEACHER_RESOURCE_POOL=False 시 특이사항](#4-teacher_resource_poolfalse-시-특이사항)
5. [Resume 방법](#5-resume-방법)
6. [OOM 해결 방법](#6-oom-해결-방법)
7. [최종 스크립트 수정 예시](#7-최종-스크립트-수정-예시)

---

## 1. 에러 발생 위치와 원인

`Variable._execution_engine.run_backward`는 PyTorch의 backward pass 진입점이다.
이 위치에서 OOM이 발생하는 것은 **FSDP backward 중 GPU 메모리 부족**을 의미한다.

### backward 시점 GPU 메모리 점유 항목

| 항목 | 설명 |
|---|---|
| FSDP parameter shard 재로드 | `param_offload=True`여도 backward 중 layer별 CPU→GPU 재로드 필요 |
| gradient buffer | 재로드된 parameter와 동일 크기 |
| activation (gradient checkpointing 재계산) | checkpointing 구간에서 activation 재계산 |
| vLLM KV cache | `free_cache_engine=True`(기본값)이면 학습 중 이미 해제됨 |

backward OOM의 직접 원인은 **activation + gradient 크기** 이므로,
`ppo_max_token_len_per_gpu` 또는 `ppo_micro_batch_size_per_gpu`를 줄이는 것이 핵심 해결책이다.

---

## 2. FSDP 학습 코드 흐름

on-policy KD에서 `update_actor` 호출 시 실제 backward가 발생하는 경로:

```
ray_trainer.py:1567       marked_timer("update_actor")
      │
      ▼
ray_trainer.py:1211       _update_actor(batch)
      │  batch에 distillation_use_topk, mini_batch_size 등 메타 추가 후
      │  → self.actor_rollout_wg.update_actor(batch_td)
      │
      ▼
engine_workers.py:642     ActorRolloutRefWorker.update_actor()
      │  → self.actor.train_mini_batch(data)
      │
      ▼
engine_workers.py:236     TrainingWorker.train_mini_batch()
      │  mini_batch_size 기준으로 iterator 생성
      │  루프: → self.train_batch(mini_batch_td)
      │
      ▼
engine_workers.py:326     TrainingWorker.train_batch()
      │  → self.engine.train_batch(data, loss_function=self.loss_fn)
      │
      ▼
engine/base.py:112        BaseEngine.train_batch()
      │  → self.forward_backward_batch(data, loss_fn, forward_only=False)
      │
      ▼
engine/fsdp/transformer_impl.py:591   forward_backward_batch()
      │  prepare_micro_batches()로 micro-batch 분할
      │  루프:
      │    → self.forward_step(micro_batch, loss_fn)
      │    → loss.backward()    ← ★ OOM 발생 지점
      │
      ▼
engine/fsdp/transformer_impl.py:1136  forward_step()
      │  prepare_model_inputs()   → FSDP forward
      │  prepare_model_outputs()  → log_probs 계산
      │  loss_function(model_output, data)  → distillation_ppo_loss()
      │  return loss, output
```

### loss function 등록 위치

`engine_workers.py:572-580`:

```python
if self.distillation_enabled:
    self.loss_fn = partial(
        distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
    )
else:
    self.loss_fn = partial(ppo_loss, config=actor_config)
self.actor.set_loss_fn(self.loss_fn)
```

---

## 3. micro_batch에 들어있는 값

`forward_step()` 진입 시점에 `micro_batch`에 들어있는 텐서들.
`use_remove_padding=True`(기본값) 기준이며, no-padding NestedTensor 형식이다.

### 텐서 데이터

| 키 | shape | dtype | 출처 |
|---|---|---|---|
| `input_ids` | `NestedTensor (bsz, j1)` | int64 | rollout |
| `position_ids` | `NestedTensor (bsz, j1)` | int64 | rollout |
| `loss_mask` | `NestedTensor (bsz, j1)` | float/bool | rollout |
| `old_log_probs` | `NestedTensor (bsz, j1)` | float32 | rollout (student) |
| `advantages` | `NestedTensor (bsz, j1)` | float32 | advantage 계산 |
| `teacher_logprobs` | `NestedTensor (bsz, j1)` | float32 | teacher vLLM |

> `teacher_ids`는 **k1 loss mode에서는 존재하지 않는다**. `forward_kl_topk` mode (`use_topk=True`)일 때만 shape `(bsz, j1, topk)`로 포함된다.

### non-tensor 메타데이터

| 키 | 값 | 의미 |
|---|---|---|
| `temperature` | float | rollout 온도 |
| `max_token_len_per_gpu` | int | dynamic bsz 상한 |
| `batch_num_tokens` | int | global token 수 (loss normalization) |
| `use_dynamic_bsz` | bool | dynamic bsz 사용 여부 |
| `distillation_use_topk` | bool | topk loss 여부 |
| `calculate_entropy` | bool | entropy 계산 여부 |

### forward → loss 계산 흐름 (k1 + use_policy_gradient=True 기준)

```python
# transformer_impl.py:1136 forward_step() 내부
raw_output = self.module(**model_inputs)     # FSDP forward
model_output = prepare_model_outputs(...)    # log_probs 추출

# distillation/losses.py:341 compute_distillation_loss_reverse_kl_estimator()
student_log_probs  = model_output["log_probs"]        # (bsz, j1) — gradient 흐름 O
teacher_log_probs  = micro_batch["teacher_logprobs"]  # (bsz, j1) — gradient 없음

distillation_losses = kl_penalty(
    logprob=student_log_probs,
    ref_logprob=teacher_log_probs,
    kl_penalty="k1"   # student - teacher
)  # → (bsz, j1)

# losses.py:261  use_policy_gradient=True → policy gradient로 변환
distillation_loss, _ = policy_loss_fn(
    old_log_prob  = micro_batch["old_log_probs"],
    log_prob      = student_log_probs,              # gradient 흐름 O
    advantages    = -distillation_losses.detach(),  # gradient 차단
    response_mask = micro_batch["loss_mask"],
)

loss.backward()   # ← OOM 발생 지점
```

---

## 4. TEACHER_RESOURCE_POOL=False 시 특이사항

`TEACHER_RESOURCE_POOL=False`는 teacher를 **standalone 모드**로 실행한다.

### standalone 모드에서 sleep/wake가 no-op

`vllm_async_server.py:565-577`:

```python
elif self.rollout_mode == RolloutMode.STANDALONE:
    logger.info("skip wake_up in standalone mode")   # 아무것도 안 함

elif self.rollout_mode == RolloutMode.STANDALONE:
    logger.info("skip sleep in standalone mode")     # 아무것도 안 함
```

`TeacherModelManager.compute_logprobs()`가 `wake_up()` / `sleep()`을 호출해도,
standalone 모드에서는 **GPU 메모리를 해제하지 않는다**.
Teacher vLLM은 학습 전 과정 내내 `gpu_memory_utilization` 만큼 GPU 메모리를 상시 점유한다.

반면 student rollout (HYBRID 모드)은 `_sleep_hybrid()` (level=2)로 학습 중 KV cache + weights를 해제한다.

### GPU 배치 시나리오

```
필요한 GPU 수 = STUDENT_WORLD_SIZE + TEACHER_WORLD_SIZE
              = 2 + 4 = 6
```

| 시나리오 | 조건 | teacher의 OOM 영향 |
|---|---|---|
| A: GPU 분리 | 노드 GPU ≥ 6 | 없음 (student/teacher GPU 별도) |
| B: GPU 공유 | 노드 GPU < 6 | **있음** (teacher가 student GPU를 상시 점유) |

시나리오 B에서는 backward OOM 발생 시 `teacher.gpu_memory_utilization`을 낮추는 것이 유효하다.
**시나리오 A에서는 teacher world size 변경이 student backward OOM에 영향을 주지 않는다.**

---

## 5. Resume 방법

기본 스크립트에 `trainer.resume_mode=disable`이 설정되어 있으므로, 이를 변경해야 한다.

### 방법 1: auto (권장)

```bash
trainer.resume_mode=auto
```

`checkpoints/${PROJECT_NAME}/${EXP_NAME}/latest_checkpointed_iteration.txt`를 읽어
**가장 최근 체크포인트**에서 자동으로 재개한다 (`ray_trainer.py:975`).

`save_freq=200`이므로 200 step 단위로 저장된 체크포인트 중 최신에서 재개된다.

### 방법 2: 특정 step 지정

```bash
trainer.resume_mode=resume_path
trainer.resume_from_path=checkpoints/${PROJECT_NAME}/${EXP_NAME}/global_step_400
```

경로 내에 `global_step_` 문자열이 포함되어야 한다 (`ray_trainer.py:982`).

### 체크포인트 경로 확인

```bash
ls checkpoints/${PROJECT_NAME}/${EXP_NAME}/
# → global_step_200/  global_step_400/  latest_checkpointed_iteration.txt
```

---

## 6. OOM 해결 방법

학습 결과(gradient 누적 동등성)를 유지하면서 메모리를 줄이는 방법이다.

### 방법 1: micro batch size 줄이기 (필수, 가장 효과적)

`ppo_micro_batch_size_per_gpu`를 줄여도 `ppo_mini_batch_size`는 유지되므로,
gradient accumulation step이 늘어날 뿐 **수학적으로 완전히 동일한 학습 결과**를 보장한다.

```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1   # 2 → 1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
# 1536 → 768
```

`use_dynamic_bsz=True`이면 `ppo_max_token_len_per_gpu`가 실질적인 메모리 상한이다 (`engine/utils.py:77-78`).

### 방법 2: student rollout KV cache 예약량 줄이기

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.2   # 0.3 → 0.2
```

student rollout (HYBRID 모드)은 `free_cache_engine=True`(기본값)이면 학습 중에는 KV cache를 해제하므로,
이 값을 줄여도 rollout throughput에만 영향을 준다.

### 방법 3: teacher gpu_memory_utilization 줄이기 (시나리오 B만 해당)

GPU가 공유되는 경우(시나리오 B)에만 의미가 있다.
Standalone 모드에서 teacher는 sleep이 없으므로 이 값이 상시 점유된다.

```bash
distillation.teacher_model.inference.gpu_memory_utilization=0.15   # 0.3 → 0.15
```

### 주의: ppo_mini_batch_size는 변경 금지

```bash
# actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ  ← 이 값은 반드시 유지
```

이 값이 바뀌면 effective batch size가 달라져 학습 결과에 직접 영향을 준다.

---

## 7. 최종 스크립트 수정 예시

```bash
# ─── 변경 사항 ───────────────────────────────────────────

STUDENT_MICRO_BATCH_SIZE_PER_GPU=1   # 2 → 1  (backward activation 절반)
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2          # 0.3 → 0.2
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ  # 변경 금지
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=auto                  # disable → auto
    trainer.log_val_generations=5
)
```

### 변경 요약

| 변경 항목 | 기존 값 | 변경 값 | 학습 결과 영향 |
|---|---|---|---|
| `trainer.resume_mode` | `disable` | `auto` | 없음 (resume) |
| `STUDENT_MICRO_BATCH_SIZE_PER_GPU` | `2` | `1` | **없음** (grad accum 2배) |
| `ppo_max_token_len_per_gpu` | `1536` | `768` | 없음 |
| `rollout.gpu_memory_utilization` | `0.3` | `0.2` | 없음 |
| `teacher.gpu_memory_utilization` | `0.3` | `0.15` | 없음 (GPU 공유 시만 적용) |
| `ppo_mini_batch_size` | — | **변경 금지** | 변경 시 학습 결과 달라짐 |

---

## 관련 코드 위치

| 위치 | 설명 |
|---|---|
| `verl/trainer/ppo/ray_trainer.py:1211` | `_update_actor()` — batch 준비 및 distillation 메타 설정 |
| `verl/trainer/ppo/ray_trainer.py:960` | `_load_checkpoint()` — resume_mode 처리 |
| `verl/workers/engine_workers.py:572` | distillation loss fn 등록 |
| `verl/workers/engine_workers.py:642` | `update_actor()` — train_mini_batch 호출 |
| `verl/workers/engine/fsdp/transformer_impl.py:591` | `forward_backward_batch()` — micro-batch 루프 + backward |
| `verl/workers/engine/fsdp/transformer_impl.py:1136` | `forward_step()` — model forward + loss 계산 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py:565` | standalone 모드에서 sleep/wake no-op |
| `verl/trainer/distillation/losses.py:221` | `distillation_loss()` — KL 계산 및 policy gradient 적용 |
| `verl/workers/engine/utils.py:59` | `prepare_micro_batches()` — dynamic bsz 분할 |
