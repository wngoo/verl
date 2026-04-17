# On-Policy KD Tensorboard 메트릭 설명

> 분석 기준 파일: `verl/trainer/distillation/losses.py`

---

## 메트릭 계산 순서 (코드 흐름)

```
1. kl_penalty() → raw k1 losses 계산                    # pre-clamp
2. distillation/abs_loss 기록                            # pre-clamp
3. distillation/loss_min, loss_max 기록                  # pre-clamp
4. loss_max_clamp 적용 (clamp)                           # ← clamp 경계
5. policy_loss_fn(advantages = -clamped_losses)          # post-clamp
6. distillation/loss 기록                                # post-clamp PPO loss
```

---

## 메트릭별 상세 설명

### `distillation/abs_loss`

**소스:** `losses.py:352` — `compute_distillation_loss_reverse_kl_estimator()` 내부

```python
"distillation/abs_loss": Metric(
    AggregationType.MEAN,
    distillation_losses[response_mask_bool].abs().mean()
)
```

- **clamp 전** raw k1 값의 절댓값 평균
- 적용 대상: `k1`, `kl`, `abs`, `mse`, `k2`, `low_var_kl`, `k3` 추정자 모드
- `k1 = exp(log q_student - log p_teacher) - 1` 은 음수가 될 수 있으므로 절댓값을 따로 기록
- **해석:** student가 teacher로부터 얼마나 벗어났는지 실제 크기를 나타내는 진단 메트릭. 낮을수록 student가 teacher 분포에 가깝다.

### `distillation/loss_min` / `distillation/loss_max`

**소스:** `losses.py:112` — `compute_distillation_loss_range()` 내부

```python
"distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
"distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
```

- **clamp 전** raw k1 토큰별 분포의 최솟값/최댓값
- response 토큰에 한정해 계산 (prompt 제외)
- **해석:** clamp 전 k1 분포의 범위를 파악하여 `loss_max_clamp` 설정 적정성을 확인하는 용도

### `distillation/loss`

**소스:** `losses.py:216` — `distillation_ppo_loss()` 내부

```python
policy_metrics["distillation/loss"] = Metric(
    value=distill_loss,
    aggregation=AggregationType.SUM
)
```

- **clamp 후** 값으로 PPO policy gradient를 계산한 결과
- `use_policy_gradient=True`(기본값)일 때:
  ```
  clamped_k1 = k1.clamp(-loss_max_clamp, +loss_max_clamp)
  advantages = -clamped_k1.detach()   # teacher에 가까울수록 높은 reward
  distill_loss = PPO_clip(old_log_prob, log_prob, advantages)
  ```
- `use_policy_gradient=False`일 때: `agg_loss(clamped_k1, response_mask)` — supervised 직접 역전파
- `AggregationType.SUM`이므로 batch token 수에 비례한 절댓값

### `actor/pg_loss`

**소스:** `verl/workers/actor/dp_actor.py:675`

- distillation loss + (task reward가 있으면 policy gradient loss)를 결합한 최종 policy loss
- `use_task_rewards=False`이면 distillation loss만 사용하므로 `distillation/loss`와 실질적으로 같은 학습 신호
- `use_task_rewards=True`이면:
  ```
  actor/pg_loss = distillation/loss * distillation_loss_coef + ppo_policy_gradient_loss
  ```

---

## clamp 전/후 요약표

| 메트릭 | clamp 적용 여부 | 집계 방식 |
|---|---|---|
| `distillation/abs_loss` | **clamp 전** (raw k1) | MEAN |
| `distillation/loss_min` | **clamp 전** (raw k1) | MIN |
| `distillation/loss_max` | **clamp 전** (raw k1) | MAX |
| `distillation/loss` | **clamp 후** (PPO loss) | SUM |
| `actor/pg_loss` | **clamp 후** (최종 loss) | SUM |

---

## reverse KL 메트릭에 대해

코드베이스에 `reverse_kl`이라는 명시적 tensorboard 키는 없다.  
`k1` estimator가 reverse KL의 단일 샘플 추정치이므로, `distillation/abs_loss`가 사실상 **reverse KL divergence 크기 추이**를 나타낸다.

```
Reverse KL = KL(q_student || p_teacher) = E_q[log q - log p]
k1(t) = exp(log q(t) - log p(t)) - 1   →   E_q[k1] = KL(q || p)  (unbiased)
```
