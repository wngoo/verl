# Entropy 계산 비교: slime vs verl (On-Policy Distillation)

On-policy distillation 수행 시 tensorboard에 로깅되는 entropy 값이 두 프레임워크에서 어떻게 계산되는지 비교 정리.

---

## 1. slime 프레임워크

### 핵심 파일 & 코드 위치

| 역할 | 파일 | 라인 |
|---|---|---|
| Entropy 계산 함수 | `slime/utils/ppo_utils.py` | 162–198 |
| 호출부 (loss 계산) | `slime/backends/megatron_utils/loss.py` | 769–771 |
| OPD 메인 로직 | `slime/rollout/on_policy_distillation.py` | 44–53 |

### 핵심 수식 — `_VocabParallelEntropy` (ppo_utils.py:162)

```python
logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)          # ① TP-wide max
normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
dist.all_reduce(normalized_sum_exp_logits, group=process_group)                  # ② TP-wide ΣEXP
softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
dist.all_reduce(sum_softmax_times_logits, group=process_group)                   # ③ TP-wide Σ(p·logit)
entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
```

**수식 의미**: `H = logsumexp(logits) - Σ(softmax(logits) · logits) = −Σ p·log p`

**핵심 포인트**:
- **Vocab-Parallel (TP)**: Megatron의 텐서 병렬 구조에 맞춰 vocab 차원이 GPU에 분할되어 있어, max / sum 단계에서 **`dist.all_reduce`로 GPU 간 동기화**가 필요. 단순 함수가 아니라 `torch.autograd.Function`으로 backward도 직접 구현.
- **Numerical stability**: `logits - max` 트릭 사용.
- **Backward 최적화**: softmax 텐서를 in-place로 재사용해 메모리 절약.

### Aggregation (loss.py:769–771)

```python
entropy = log_probs_and_entropy["entropy"]
entropy = torch.cat(entropy, dim=0)
entropy_loss = sum_of_sample_mean(entropy)        # 샘플별 평균 → 배치 평균
loss = pg_loss - args.entropy_coef * entropy_loss
```

→ **token-level entropy를 sample별로 평균낸 뒤, 샘플 간 평균**. 마스킹은 `sum_of_sample_mean` 내부에서 loss_mask로 처리.

### OPD에서의 위치

- Teacher는 **logprob만** 제공 (`on_policy_distillation.py:44-53`에서 `sample.teacher_log_probs` 저장).
- Tensorboard에 찍히는 entropy는 **student 모델**의 정책 분포 entropy.
- Distillation 신호는 advantage에 reverse-KL을 빼서 주입 (`apply_opd_kl_to_advantages`, loss.py:359).

---

## 2. verl 프레임워크

### 핵심 파일 & 코드 위치

| 역할 | 파일 | 라인 |
|---|---|---|
| Entropy 계산 함수 | `verl/utils/torch_functional.py` | 224–263 |
| Actor 호출부 | `verl/workers/actor/dp_actor.py` | 83–92, 277 |
| Distillation 로직 | `verl/trainer/distillation/losses.py` | 164–290 |
| 메트릭 로깅 | `verl/trainer/main_ppo_sync.py` | 1206–1246 |

### 핵심 수식 — `entropy_from_logits` (torch_functional.py:224)

```python
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
```

**수식 의미**: 동일하게 `H = logsumexp(logits) - Σ(p · logits) = −Σ p·log p`

**메모리 절약형 변형** (line 241, chunking):

```python
def entropy_from_logits_with_chunking(logits, chunk_size=2048):
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i:i+chunk_size].float()
        pd_chunk = F.softmax(logits_chunk, dim=-1)
        entropy[i:i+chunk_size] = logsumexp(logits_chunk, -1) - sum(pd_chunk * logits_chunk, -1)
```

→ batch 차원을 2048씩 잘라서 처리, float32 cast로 안정성 확보.

### Actor에서의 wiring (dp_actor.py:83-92)

```python
if self.config.entropy_from_logits_with_chunking:
    entropy_from_logits = verl_F.entropy_from_logits_with_chunking
else:
    entropy_from_logits = verl_F.entropy_from_logits

self.compute_entropy_from_logits = (
    torch.compile(entropy_from_logits, dynamic=True)
    if self.config.get("use_torch_compile", True)
    else entropy_from_logits
)
```

→ `torch.compile`로 가속, forward pass 중 `self.compute_entropy_from_logits(logits_rmpad)` 호출 (line 277).

### Aggregation

- `agg_loss(entropy, response_mask, loss_agg_mode)`로 처리.
- `loss_agg_mode`에 따라 `token-mean`, `seq-mean-token-mean`, `seq-mean-token-sum` 선택 가능.
- `response_mask`로 prompt 토큰은 제외하고 response 토큰만 집계.

### Distillation

- `verl/trainer/distillation/losses.py:164` (`distillation_ppo_loss`): student의 forward-KL(teacher‖student)을 top-k 방식으로 계산 (`compute_forward_kl_topk`, line 293).
- `use_policy_gradient=True`이면 negative distillation loss를 **reward로 변환**해 on-policy로 흘림 (line 256-278).
- Tensorboard에 찍히는 entropy는 마찬가지로 **student 모델**의 분포 entropy.

---

## 3. 두 프레임워크 비교

| 항목 | slime | verl |
|---|---|---|
| **수식** | `logits_max + log(Σexp(norm_logit)) − Σ(softmax·logit)` | `logsumexp(logit) − Σ(softmax·logit)` |
| **수학적 결과** | 동일 (`−Σ p·log p`) | 동일 (`−Σ p·log p`) |
| **병렬화** | **Vocab-Parallel** (Megatron TP) — `dist.all_reduce` 3회 | **단일 디바이스** — 청킹으로 메모리 절약 |
| **구현 형태** | `torch.autograd.Function` (forward/backward 직접 구현) | 단순 함수 + `torch.compile` |
| **수치 안정성** | `logits − max` 트릭 | `logsumexp`로 자동 처리 |
| **메모리 최적화** | backward에서 softmax 텐서 in-place 재사용 | chunking + float32 cast |
| **Aggregation** | `sum_of_sample_mean` (샘플 평균 후 배치 평균 고정) | `agg_loss` (token-mean/seq-mean 선택 가능) |
| **Masking** | loss_mask | response_mask |
| **대상 모델** | Student (teacher는 logprob만 제공) | Student (teacher는 top-k logprob 제공) |
| **OPD signal** | reverse-KL을 advantage에서 차감 | forward-KL(top-k) loss → reward 또는 직접 backprop |
| **백엔드** | Megatron-LM | FSDP / Megatron 모두 지원 |

### 수치적으로 같은 값이 나오는가?

**이론적으로는 동일하지만 실측 값은 미세하게 다를 수 있음**:

1. **수식 동등성**: 두 수식 모두 정확히 `H(p) = −Σ p · log p`이며 수학적으로 동일.
2. **차이의 원인**:
   - slime은 `logits − max` 후 exp를 취하므로 float overflow에 강함 / verl의 `logsumexp`도 동일한 안정성을 보장 → fp32에서는 같은 값.
   - **bf16/fp16 cast 시점, all_reduce 누적 순서, chunking 경계**에서 부동소수점 누적 오차가 다름.
   - slime의 vocab-parallel all_reduce는 GPU 수에 따라 합산 순서가 바뀌어 reproducibility가 더 미묘.
3. **Aggregation 차이가 더 큼**: 두 프레임워크에서 entropy 값을 직접 비교하려면 같은 aggregation 방식을 써야 함. slime의 `sum_of_sample_mean`(샘플 평균→배치 평균)과 verl의 `token-mean`(전체 valid 토큰 평균)은 **샘플 길이가 균일하지 않으면 다른 값**이 나옴.

### 비교할 때 권장사항

- **같은 토큰 집합** (동일한 prompt + response)에서 비교.
- **같은 aggregation** 사용 (verl에서 `token-mean`으로 맞춘 뒤, slime 쪽 mask 처리도 token-mean 형태로 측정).
- **fp32**로 cast해서 비교 (bf16 누적 오차 제거).
- TP 환경에서 slime 결과를 검증할 땐 TP=1로 두면 verl과 동일한 구현이 되어야 함 (all_reduce가 no-op).

---

## 요약

두 프레임워크 모두 정확히 같은 Shannon entropy `−Σ p log p`를 계산하지만,

- **slime**: Megatron의 텐서 병렬 환경에 맞춰 **vocab-parallel autograd Function**으로 구현 (`ppo_utils.py:162`)
- **verl**: 단일 디바이스 기준 단순 함수 + 청킹 + `torch.compile` (`torch_functional.py:224`)

실제 로깅 값을 비교하려면 **aggregation 방식(sample-mean vs token-mean)**과 **TP/precision 설정**을 통일하는 것이 가장 중요.
