# On-Policy Distillation: `distillation/loss` vs `abs_loss` 해석

> 분석 기준 파일: `verl/trainer/distillation/losses.py`
> 대상 설정: `loss_mode=k1`, `use_policy_gradient=True`, `use_task_rewards=False`

---

## TL;DR

| 질문 | 답 |
|---|---|
| backward에 실제로 들어가는 건? | **`distillation/loss`** (= `distill_loss` 텐서 그 자체) |
| 학습량(gradient 크기)을 잘 보여주는 건? | **`abs_loss`** (≈ advantage 분산의 proxy) |
| student↔teacher 거리를 보여주는 건? | **`abs_loss`** |
| `distillation/loss`가 0에 가까워지면 학습 멈춤? | **아님.** advantage mean이 0에 수렴해도 분산이 있으면 학습 계속됨 |

`abs_loss` 가 단조 감소하면 학습은 잘 진행 중. `distillation/loss`의 절대값/부호로 학습 잘됨을 판정하면 안 된다.

---

## 1. 두 메트릭의 정확한 정의

`k1` + `use_policy_gradient=True` 조합에서:

| 메트릭 | 시점 | 값 | 집계 |
|---|---|---|---|
| `distillation/abs_loss` | clamp 전, raw | `|log q_student(t) − log p_teacher(t)|` 평균 | MEAN |
| `distillation/loss` | clamp 후, PPO surrogate | `PPO_clip(old_log_prob, log_prob, A)` | **SUM** |

advantage 정의: `A = −clamped_k1` (`losses.py:264`).

- `abs_loss` → student와 teacher의 **실제 거리**(reverse KL의 단일 샘플 크기).
- `loss` → 그 거리를 advantage로 박은 **PPO 학습 신호**. 거리 자체가 아님.

---

## 2. 무엇이 backward 되는가

`losses.py:204-218`:

```python
distill_loss, distill_metrics = distillation_loss(...)
policy_loss, policy_metrics = ppo_loss(...)
if not distillation_loss_config.use_task_rewards:
    policy_loss = 0.0

policy_loss += distill_loss * distillation_loss_coef    # ← backward 대상
policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)
```

**`distill_loss` 텐서가 그대로 `policy_loss`에 더해져 backward 됨**, 그리고 동일 텐서 값이 `distillation/loss` 메트릭으로 기록됨. 즉 backward 되는 값과 메트릭은 동일 객체.

`abs_loss`는 backward 경로에 없음. 순수 진단용.

---

## 3. 왜 `distillation/loss`로 학습 잘됨을 판정하면 안 되는가

PPO surrogate loss는 supervised loss와 성질이 다름. `train_batch_size == ppo_mini_batch_size` (mini_epoch=1) → forward 시점에 `ratio ≈ 1`. clip 미발동 구간에서:

```
distill_loss ≈ −E[A_t] = E[k1_t] = E[log q_student − log p_teacher]
```

핵심 관찰:

- **gradient 크기는 `E[A]`가 아니라 `Var(A)`에 좌우됨.**
- `A_t`가 토큰마다 +/− 섞여 있으면 평균은 0에 가까워도 token-level `A_t · ∇log π`는 살아 있음.
- 따라서 `distillation/loss`가 -0.2에서 횡보해도 gradient는 충분히 클 수 있다.

비유: GRPO/PPO에서 advantage를 normalize하면 mean=0이지만 학습은 잘 됨. 같은 원리. **surrogate loss 값 ≠ 학습량**.

---

## 4. 사례 해석: `abs_loss` ↓, `loss` -0.4 → -0.1 → -0.2 횡보

### (a) `abs_loss` 꾸준히 감소
가장 신뢰할 수 있는 진짜 신호. student 분포가 teacher에 가까워지고 있음.

### (b) `loss` -0.4 → -0.1 (절댓값 감소)
abs_loss 감소를 PPO 쪽에서 비춘 그림자. advantage `−k1`의 평균이 0에 가까워지는 자연스러운 수렴 거동. 같은 사건을 두 메트릭이 다르게 보여주는 것뿐.

### (c) -0.1 → -0.2 후 횡보
가능한 원인 (영향 큰 순):

1. **SUM aggregation의 batch 민감도**
   `distillation/loss`는 SUM, `abs_loss`는 MEAN. response 길이 분포가 step마다 바뀌면 SUM은 그에 비례해 출렁임. abs_loss가 monotonic해도 loss는 진동 가능.
2. **On-policy 분포 이동**
   student가 갱신될수록 rollout 분포 자체가 변함. 새로 탐색하는 토큰 영역에서 일시적으로 `|k1|`이 커지는 토큰 발생 → 평균 거리는 줄지만 advantage 분산이 늘어 PPO 신호가 다시 강해 보임.
3. **k1 estimator의 high variance**
   reverse KL의 single-sample 추정자라 mean(k1) 자체가 본질적으로 진동.
4. **`loss_max_clamp=10` 영향은 낮음**
   abs_loss가 작아진 단계에서는 clamp 발동 토큰 비율이 줄어 영향 거의 없음.

---

## 5. 함께 점검하면 좋은 메트릭

| 메트릭 | 무엇을 보는가 |
|---|---|
| `actor/grad_norm` | 진짜 학습이 진행 중인지의 결정적 지표 (0 근처면 정체) |
| `actor/distillation/loss_min`, `loss_max` | clamp 경계에 닿는 토큰이 줄어드는지 |
| rollout `response_length` 평균 추이 | `loss`(SUM) 횡보가 batch 구성 효과인지 판별 |
| val 품질 지표 / `log_val_generations` | 결국 distillation의 끝 목표. abs_loss와 같은 방향이면 안심 |

---

## 6. 결론

- **backward 되는 건 `distillation/loss`가 맞다.** 그래서 "학습에 반영되는 신호" 자체로 보면 이 메트릭이 정답.
- 하지만 그 **값의 부호·절댓값을 보고 학습 잘됨을 판정하면 잘못된 결론**에 도달함. PPO surrogate는 gradient 만들기 위한 계산일 뿐.
- **학습 진행/수렴 판단은 `abs_loss` + `grad_norm` + val 메트릭** 조합으로 한다.
- `abs_loss`가 단조 감소하는 한 `distillation/loss` 횡보는 **수렴의 결과이지 정체의 원인이 아니다**.
