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
