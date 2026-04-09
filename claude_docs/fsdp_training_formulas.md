---
date: 2026-04-09
topic: FSDP/ZeRO-3 학습 핵심 공식 모음 (배치 크기, 메모리, TP 관계)
tags: [fsdp, zero3, batch_size, memory, tp, gradient_accumulation, linear_scaling, verl]
---

# FSDP 학습 핵심 공식 모음

---

## 1. FSDP에서 TP의 역할 (중요)

```
FSDP  = ZeRO-3 Data Parallelism → TP=1 고정, PP=1 고정
          data_parallel_size = total_gpus

Megatron = data_parallel_size = total_gpus / (TP × PP)
```

**verl에서 TP 설정은 rollout(vLLM/SGLang) 전용이다. actor FSDP 학습에는 관여하지 않는다.**

---

## 2. ZeRO-3 메모리 공식

모델 파라미터 수를 `N`, GPU 수를 `dp`라 할 때:

```
# 전체 메모리 부담 (ZeRO 없을 때)
weights     : N × 2 bytes  (BF16)
gradients   : N × 2 bytes  (BF16)
optimizer   : N × 8 bytes  (FP32 master + Adam m, v)
총합        : N × 12 bytes

# ZeRO-3 적용 후 GPU당 모델 관련 메모리
per_gpu_model = (N × 12 bytes) / dp
```

### 예시

| 모델 | dp (GPUs) | GPU당 모델 메모리 |
|---|---|---|
| 4B | 32 | 4B × 12 / 32 = **1.5 GB** |
| 4B | 8  | 4B × 12 / 8  = **6 GB** |
| 70B | 64 | 70B × 12 / 64 = **13 GB** |

---

## 3. Activation 메모리 공식

`bs` = micro_batch_size, `L` = seq_len, `H` = hidden_size, `nl` = num_layers

```
# Gradient Checkpointing 미적용
A ≈ bs × L × H × 2bytes × nl × 34

# Gradient Checkpointing 적용 (재계산 방식)
A ≈ bs × L × H × 2bytes × nl × 2
```

### 128K context 예시 (4B: H=2048, nl=28, micro_bs=1)

```
미적용: 1 × 131072 × 2048 × 2 × 28 × 34 ≈ 510 GB  ← 불가능
적용:   1 × 131072 × 2048 × 2 × 28 × 2  ≈  30 GB  ← 가능
```

→ **긴 context 학습 시 gradient checkpointing은 필수**

---

## 4. Global Batch Size 공식 (verl PPO/GRPO)

```
global_batch_size = micro_batch_size × dp_size × gradient_accumulation_steps

# 역산: accumulation steps
gradient_accumulation = ppo_mini_batch_size / (micro_batch_size × dp_size)

# PPO update 횟수 (per rollout)
num_updates_per_iter = global_batch_size / ppo_mini_batch_size
```

### verl config 대응

```yaml
data:
  train_batch_size: 512          # global_batch_size (1 iteration당 rollout 시퀀스 수)

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128     # 1회 gradient update에 사용하는 시퀀스 수
    ppo_micro_batch_size: 4      # GPU 1개가 한 번에 처리하는 시퀀스 수
    # gradient_accumulation = 128 / (4 × 32 GPUs) = 1
```

---

## 5. 정합성 체크 (설정 오류 방지)

```python
# 이 조건이 모두 성립해야 함
assert ppo_mini_batch_size % (micro_batch_size × dp_size) == 0
assert global_batch_size % ppo_mini_batch_size == 0
assert global_batch_size % dp_size == 0
```

### 128K context에서의 현실적 제약

```
micro_batch_size = 1  # 128K에서 사실상 강제 (activation 메모리 한계)

gradient_accumulation = ppo_mini_batch_size / dp_size
# 예: dp=32, mini_bs=32 → accumulation=1 (권장: accumulation=1)
#     dp=32, mini_bs=128 → accumulation=4
```

---

## 6. Linear Scaling Rule (GPU 수 변경 시 LR 조정)

```
# Batch size 스케일링
new_global_bs = original_bs × (new_gpus / original_gpus)

# LR 스케일링 방법 2가지
linear rule : new_lr = original_lr × (new_bs / original_bs)
sqrt rule   : new_lr = original_lr × sqrt(new_bs / original_bs)  ← 대형 모델 권장
```

### 예시: 8 GPUs → 64 GPUs (8배 증가)

```
global_bs : 256 → 2048
LR (sqrt) : 1e-4 → 1e-4 × √8 ≈ 2.83e-4
LR (linear): 1e-4 → 8e-4  (너무 크면 불안정)
```

---

## 7. Rollout TP와 학습 GPU 수의 관계

```
# Rollout vLLM 인스턴스 수
num_rollout_instances = student_gpus / rollout_tp_size

# 학습 유효 DP (FSDP는 항상 전체 GPU)
train_dp_size = student_gpus

# Actor + Rollout이 같은 GPU를 공유(colocate)해도
# rollout_tp_size는 train_dp_size에 영향 없음
```

---

## 8. 실제 설정 예시 (4B student, 32 GPUs, 128K)

```
dp_size = 32
micro_batch_size = 1       (128K 강제)
ppo_mini_batch_size = 32   (dp와 같게 → accumulation=1)
global_batch_size = 128    (4회 mini-batch update)

gradient_accumulation = 32 / (1 × 32) = 1

tokens_per_step = 128 × 131,072 ≈ 16.8M tokens/iter
```

```yaml
data:
  train_batch_size: 128
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: true          # 128K에서 길이 편차 있을 때 유리
    ppo_max_token_len_per_gpu: 131072
```

---

## 9. 요약 치트시트

| 변수 | 공식 |
|---|---|
| ZeRO-3 GPU당 모델 메모리 | `(N × 12 bytes) / dp_size` |
| Activation (checkpointing 적용) | `bs × L × H × 2B × layers × 2` |
| Gradient accumulation steps | `mini_bs / (micro_bs × dp_size)` |
| LR scaling (GPU 증가 시) | `lr × sqrt(new_bs / old_bs)` |
| Rollout 인스턴스 수 | `student_gpus / rollout_tp` |
| 정합성 조건 | `mini_bs % (micro_bs × dp) == 0` |

---

## 관련 문서

- verl batch 파라미터 상세: `claude_docs/verl_batch_config_explained_2.md`
- GPU 파라미터 설정 전반: `claude_docs/gpu_settings_guide.md`
- 128K OPD 설정 예시: `claude_docs/gemma4_opd_128k_tp_pp_h100x64.md`
