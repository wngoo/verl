# ppo_micro_batch_size_per_gpu & ppo_max_token_len_per_gpu 상세 설명

## 배경

Gemma4 on-policy KD 학습 시 FlashAttention head_dim 제한(>256)으로 인해 `attn_implementation=sdpa`로
전환하면 OOM이 발생한다. TP=8(최대), batch_size=256, context_length 고정 조건에서 메모리를 줄이기 위해
이 두 파라미터를 조정하는 방법을 검토한다.

---

## 배치 크기 계층 구조

```
train_batch_size (256)          ← 전체 rollout에서 수집된 샘플 수 (고정)
  └─ ppo_mini_batch_size (256)  ← 1번의 PPO 파라미터 업데이트에 쓰이는 샘플 수
       └─ micro_batch            ← 실제로 GPU에 한 번에 올라가는 단위 (gradient accumulation)
```

---

## `ppo_micro_batch_size_per_gpu`

**역할**: `use_dynamic_bsz=False`일 때, mini_batch를 몇 개의 샘플씩 잘라서 forward/backward를
수행할지 결정한다. Gradient accumulation 단위다.

```python
# verl/workers/actor/dp_actor.py:568-591
self.gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)

for micro_batch in micro_batches:
    loss = forward(micro_batch)
    loss_scale = 1 / gradient_accumulation  # gradient를 accumulation 수로 나눠서 정규화
    (loss * loss_scale).backward()          # gradient 누적

optimizer.step()  # mini_batch 전체에 대한 1번의 업데이트
```

**학습 결과에 영향 없음**: `ppo_micro_batch_size_per_gpu`를 줄여도, gradient accumulation을 더 많이
하여 동일한 mini_batch 전체에 대한 gradient가 optimizer.step()에 적용된다. 수학적으로 동등하다.

```
micro_batch=2: [step 2샘플 → step 2샘플 → ... → optimizer.step()] ← mini_batch 전체와 동일
micro_batch=1: [step 1샘플 → step 1샘플 → ... → optimizer.step()] ← 동일한 결과
```

---

## `ppo_max_token_len_per_gpu`

**역할**: `use_dynamic_bsz=True`일 때만 사용된다. 고정된 샘플 수 대신 **토큰 수 기준**으로
micro_batch를 나눈다.

```python
# verl/workers/actor/dp_actor.py:562-566
if use_dynamic_bsz:
    max_token_len = ppo_max_token_len_per_gpu * ulysses_sp_size
    micro_batches = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
    # 각 micro_batch의 총 토큰 수 ≤ max_token_len 이 되도록 자동 분할
```

짧은 시퀀스는 한 micro_batch에 여러 샘플, 긴 시퀀스는 적은 샘플이 들어가 GPU 활용률을 높인다.

**학습 결과에 거의 영향 없음**: loss scale이 샘플 수 기준으로 정규화된다.

```python
# use_dynamic_bsz=True일 때 (dp_actor.py:589)
loss_scale = micro_batch_size / ppo_mini_batch_size  # 샘플 수 기준 정규화

# 예: ppo_mini_batch_size=256이고 micro_batch에 샘플 3개가 들어왔으면
loss_scale = 3 / 256
```

`ppo_max_token_len_per_gpu`를 줄이면 micro_batch당 샘플 수가 줄어들지만, mini_batch 전체 합산으로
gradient를 계산하므로 이론적으로 동등하다. 부동소수점 연산 순서에 따른 미세한 수치 차이는 있을 수 있다.

---

## 비교 요약

| 파라미터 | 적용 조건 | 학습 결과 영향 | 변경 가능 여부 |
|---|---|---|---|
| `ppo_micro_batch_size_per_gpu` | `use_dynamic_bsz=False` | **없음** | ✅ 자유롭게 줄일 수 있음 |
| `ppo_max_token_len_per_gpu` | `use_dynamic_bsz=True` | **거의 없음** (부동소수점 미세 차이) | ✅ 줄일 수 있음 |

---

## OOM 해결을 위한 권장 설정

```bash
STUDENT=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1   # 줄이기 (예: 2 → 1)
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048   # 시퀀스 길이에 맞게 조정
    actor_rollout_ref.actor.use_dynamic_bsz=True             # dynamic 사용 권장
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.model.enable_activation_offload=True
)

export PYTORCH_ALLOC_CONF=expandable_segments:True
```

`use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu` 조합이 메모리 효율이 가장 좋다.

---

## 관련 파일

- `verl/workers/actor/dp_actor.py:547-616` — micro_batch 분할 및 gradient accumulation 로직
- `verl/workers/config/actor.py:145,148` — `ppo_micro_batch_size_per_gpu`, `ppo_max_token_len_per_gpu` 기본값
- `verl/utils/seqlen_balancing.py` — `prepare_dynamic_batch` / `rearrange_micro_batches` 구현

---

## 신규 engine 구현에서의 정규화 방식 (`use_legacy_worker_impl=disable`)

> 스크립트에 `trainer.use_legacy_worker_impl=disable`이 설정된 경우 아래 구현이 사용된다.

### batch_num_tokens 기반 정규화

`forward_backward_batch()` (`engine/fsdp/transformer_impl.py:595-605`)에서
**micro-batch로 나누기 전에** 전체 mini-batch의 유효 token 수를 미리 계산한다:

```python
# micro-batch 분할 전에 global token 수 계산
batch_num_tokens = data["loss_mask"].sum()
torch.distributed.all_reduce(batch_num_tokens, ...)   # 전체 DP rank 합산
tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())

# 그 다음에 micro-batch 분할
micro_batches, indices = prepare_micro_batches(data, ...)

for micro_batch in micro_batches:
    loss, meta = forward_step(micro_batch, loss_fn)
    loss.backward()
```

각 micro-batch의 loss는 `agg_loss()` (`core_algos.py:1168-1173`)에서 이 값으로 나뉜다:

```python
# token-mean 모드 (기본값)
loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
```

`batch_num_tokens`는 mini-batch 전체 기준이므로, micro-batch를 어떻게 나누든 gradient 크기가 동일하다.
이것이 legacy 구현(`1 / gradient_accumulation` 스케일)과 동일한 보장을 제공하는 이유다.

### OOM 발생 시 resume하면서 파라미터 변경 가능 여부

| 파라미터 | 역할 | resume 시 변경 가능 여부 | 근거 |
|---|---|---|---|
| `ppo_micro_batch_size_per_gpu` | gradient accumulation 단위 (메모리 엔지니어링) | **가능** | `batch_num_tokens`로 정규화하므로 micro-batch 크기 무관 |
| `ppo_max_token_len_per_gpu` | dynamic bsz 상한 (메모리 엔지니어링) | **가능** | 동일한 이유 |
| `ppo_mini_batch_size` | effective batch size (학습 하이퍼파라미터) | **변경 금지** | 변경 시 `batch_num_tokens` 자체가 달라져 gradient magnitude 변경됨 |

`ppo_mini_batch_size`만 고정하면, 나머지 두 파라미터는 "같은 gradient를 어떻게 쪼개서 계산하느냐"의
문제이므로 학습 결과에 영향을 주지 않는다.

### 관련 파일 (신규 engine)

- `verl/workers/engine/fsdp/transformer_impl.py:591` — `forward_backward_batch()` — batch_num_tokens 계산 위치
- `verl/workers/utils/losses.py:58` — `ppo_loss()` — global_batch_info에 batch_num_tokens 주입
- `verl/trainer/ppo/core_algos.py:1138` — `agg_loss()` — token-mean 정규화 구현
- `verl/workers/engine/utils.py:59` — `prepare_micro_batches()` — dynamic bsz micro-batch 분할
