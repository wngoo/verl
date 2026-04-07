# verl 배치 설정 완전 가이드

---
date: 2026-04-08
topic: verl의 micro_batch_size, token_len, dynamic_bsz 설정의 의미와 동작 원리
tags: [verl, batch, dynamic_bsz, micro_batch, token_len, fsdp, gradient_accumulation]
---

## 개요

verl에서 학습 배치를 제어하는 핵심 파라미터 3가지:
- `ppo_micro_batch_size_per_gpu`
- `ppo_max_token_len_per_gpu`
- `use_dynamic_bsz`

이 파라미터들은 `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`에서 다음과 같이 설정된다:

```bash
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
# = 2 * (256 + 512) = 1536
USE_DYNAMIC_BSZ=True
```

---

## 배치 계층 구조

```
global train batch (TRAIN_PROMPT_BSZ 샘플)
    └── per-GPU shard (TRAIN_PROMPT_BSZ / GPU 수)
            └── mini-batch (ppo_mini_batch_size)
                    └── micro-batch ← 실제 forward/backward 단위
```

---

## ppo_micro_batch_size_per_gpu

### 왜 per_gpu 단위인가?

FSDP는 모델 파라미터를 GPU 간 분산하지만, **데이터는 각 GPU가 독립적으로 처리**한다.
- GPU0은 자신의 데이터 shard만 처리
- GPU 간 통신 없이 각 GPU가 독립적으로 micro-batch 크기 결정

### use_dynamic_bsz=False일 때 (정적)

```python
# dp_actor.py:568-571
self.gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
# 예: 128 // 2 = 64번 gradient 누적
micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)
# 항상 2샘플씩 균등 분할
```

- 한 번에 2샘플 forward/backward → 64번 누적 후 optimizer step
- **목적**: GPU 메모리 제어, OOM 방지

### use_dynamic_bsz=True일 때 (동적)

`ppo_micro_batch_size_per_gpu`는 직접 사용되지 않고,
`ppo_max_token_len_per_gpu` 계산의 기준값으로만 활용된다:
```bash
STUDENT_MAX_TOKEN_LEN_PER_GPU = STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH)
# = 최대 길이 샘플 2개가 들어갈 수 있는 토큰 예산
```

---

## ppo_max_token_len_per_gpu

### 왜 샘플 수가 아닌 토큰 수로 제어하는가?

LLM 학습에서 **메모리 사용량은 총 토큰 수에 비례**한다. 고정 샘플 수로 자르면:
- 짧은 샘플: GPU 낭비
- 긴 샘플: OOM 위험

### use_dynamic_bsz=True일 때 동작 (seqlen_balancing.py)

```python
total_seqlen = seq_len_effective.sum()
num_micro_batches = ceildiv(total_seqlen, max_token_len)  # 필요한 micro-batch 수 자동 계산

# FLOPs 기반 부하 균형 (24576*L + L²)
workloads = calculate_workload(seq_len_effective)
micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches)
```

**예시 (max_token_len=1536):**

| 시퀀스 길이 | micro-batch당 샘플 수 |
|------------|----------------------|
| 768 tokens (최대) | 2개 |
| 150 tokens (짧음) | ~10개 |
| 혼재 | FLOPs 균등 배분 |

### DP 랭크 간 동기화

```python
# 모든 GPU가 동일한 수의 micro-batch를 처리하도록 all_reduce MAX
num_micro_batches = torch.tensor([num_micro_batches])
dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX)
```

---

## use_dynamic_bsz

### False vs True 비교

| 비교 항목 | False | True |
|-----------|-------|------|
| OOM 안전성 | 긴 시퀀스 → OOM 위험 | 토큰 총량 상한으로 안전 |
| GPU 활용률 | 짧은 시퀀스 → 낭비 | 자동 최대화 |
| 학습 품질 | 동일 | 동일 |
| PrefixGrouper | 호환 | 비호환 (dp_actor.py:135) |

### Loss scale 보정

```python
# dp_actor.py:588-591
if use_dynamic_bsz:
    # micro-batch마다 샘플 수가 다름 → 비율로 보정
    loss_scale_factor = response_mask.shape[0] / ppo_mini_batch_size
else:
    loss_scale_factor = 1 / gradient_accumulation
```

수학적으로 전체 mini-batch를 한 번에 처리한 것과 동일한 gradient를 보장한다.

### 언제 False를 써야 하나?

- `PrefixGrouper`를 사용할 때 (공유 prefix 최적화)
- 시퀀스 길이가 매우 균일한 데이터 (차이 미미)
- 그 외 일반 상황에서는 True 권장

---

## 관련 파일/코드

- `verl/workers/actor/dp_actor.py` — 실제 micro-batch 분기 로직
- `verl/utils/seqlen_balancing.py` — `rearrange_micro_batches`, `prepare_dynamic_batch`
- `verl/trainer/config/actor/actor.yaml` — 파라미터 정의
- `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` — 예제 설정
