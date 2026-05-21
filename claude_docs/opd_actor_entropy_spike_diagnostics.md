# On-Policy Distillation: `actor/entropy` 급등 진단 가이드

> 대상: verl OPD (loss_mode=k1, use_policy_gradient=True, use_task_rewards=False 기준)
> 목적: tensorboard `actor/entropy`가 갑자기 튈 때 원인을 분리하기 위한 보조 메트릭과 추가 위치 정리

---

## 1. 가능한 원인 가설

OPD에서 `actor/entropy`가 급등하는 시나리오는 크게 네 가지로 나뉩니다.

1. **분포 collapse / 평탄화** — student 분포가 전체적으로 평평해지는 방향으로 학습됨 (weight 발산, LR 과대, optimizer 폭주 동반).
2. **Outlier 토큰** — 일부 토큰(EOS 인접, multi-turn 경계, masking 버그)만 entropy가 폭주.
3. **학습 신호 saturate** — `loss_max_clamp` 경계에 자주 닿아 PPO advantage가 ±clamp의 binary signal처럼 동작, gradient noise 증가.
4. **Off-policy gap** — vllm rollout 정책과 FSDP train 정책 격차가 커져 student가 자기가 만들지 않은 분포에 대해 학습.

평균 한 점(`actor/entropy_loss`)만 보면 이 네 가지를 구분할 수 없습니다. 분포·꼬리·관련 신호를 함께 찍어야 합니다.

---

## 2. 이미 있는 메트릭 (먼저 동시 시점 확인)

| 메트릭 | 위치 | 의미 |
|---|---|---|
| `actor/grad_norm` | `verl/workers/actor/dp_actor.py:679` 등 | 가중치 폭주 1차 신호 |
| `distillation/loss_min`, `distillation/loss_max` | `verl/trainer/distillation/losses.py:114-116` | k1 raw 범위 — clamp boundary 접근 여부 |
| `distillation/abs_loss` | `verl/trainer/distillation/losses.py:352` | reverse-KL 크기 (student↔teacher 거리) |
| `distillation/pg_clipfrac`, `distillation/ppo_kl` | `verl/trainer/distillation/losses.py:270` (rename 후) | PPO clip 빈도, on-policy 위반 |
| `rollout_corr/*` | `verl/trainer/ppo/rollout_corr_helper.py` (호출 `dp_actor.py:637-644`) | vllm rollout ↔ FSDP train 정책 격차 |

**`distillation/loss_max`가 `loss_max_clamp` 값에 자주 닿고 + `pg_clipfrac` ↑ + `ppo_kl` ↑** 이면 가설 (3) 신호 saturate가 거의 확정입니다.

---

## 3. 새로 추가할 메트릭

### (A) Entropy 분포 (mean → quantile)

`actor/entropy_loss`는 mean만 찍히므로 "outlier 한두 개"와 "전체 평탄화"가 구분되지 않음. p50/p90/p99/max를 같이 찍어야 함.

**추가 위치**: `verl/workers/utils/losses.py:116-122` (engine path; `use_legacy_worker_impl=disable` 경로)

```python
# losses.py, ppo_loss() 내부 — 기존 entropy_loss 계산 직후
if entropy is not None:
    entropy_loss = agg_loss(
        loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    entropy_coeff = config.entropy_coeff
    policy_loss -= entropy_coeff * entropy_loss
    metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # === ADD: entropy distribution diagnostics ===
    with torch.no_grad():
        ent_resp = entropy[response_mask]
        if ent_resp.numel() > 0:
            ent_f = ent_resp.float()
            qs = torch.quantile(ent_f, torch.tensor([0.5, 0.9, 0.99], device=ent_f.device))
            metrics["actor/entropy_p50"] = Metric(value=qs[0], aggregation=AggregationType.MEAN)
            metrics["actor/entropy_p90"] = Metric(value=qs[1], aggregation=AggregationType.MEAN)
            metrics["actor/entropy_p99"] = Metric(value=qs[2], aggregation=AggregationType.MEAN)
            metrics["actor/entropy_max"] = Metric(value=ent_f.max(), aggregation=AggregationType.MAX)
            metrics["actor/entropy_std"] = Metric(value=ent_f.std(), aggregation=AggregationType.MEAN)
```

**해석**
- entropy_mean과 entropy_p50/p90이 같이 오르면 → **분포 전체 평탄화** (가설 1, weight 발산 동반 가능성).
- entropy_max/p99만 오르면 → **일부 outlier 토큰** (가설 2, masking·EOS 인접 의심).

### (B) Student-Teacher log-prob gap 분포 + clamp 도달 비율

`k1 = exp(log_q - log_p) - 1` 이므로 `log_q - log_p` 분포가 학습 신호의 원천. clamp 이후 학습 신호가 어디서 나오는지 보려면 raw gap + clamp boundary 도달 비율을 함께 찍어야 함.

**추가 위치**: `verl/trainer/distillation/losses.py:325-354` `compute_distillation_loss_reverse_kl_estimator()` 내부

```python
# compute_distillation_loss_reverse_kl_estimator() 내부, 기존 metrics 직전
student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
response_mask_bool = data["response_mask"].bool()

# === ADD: log-prob gap + clamp saturation diagnostics ===
gap = (student_log_probs - teacher_log_probs)[response_mask_bool].float()
metrics_extra = {
    "distillation/logprob_gap_mean": Metric(AggregationType.MEAN, gap.mean()),
    "distillation/logprob_gap_p99":  Metric(AggregationType.MEAN, torch.quantile(gap, 0.99)),
    "distillation/logprob_gap_p01":  Metric(AggregationType.MEAN, torch.quantile(gap, 0.01)),
    "distillation/student_logprob_min": Metric(AggregationType.MIN, student_log_probs[response_mask_bool].min()),
    "distillation/teacher_logprob_min": Metric(AggregationType.MIN, teacher_log_probs[response_mask_bool].min()),
}
loss_config: DistillationLossConfig = distillation_config.distillation_loss
if loss_config.loss_max_clamp is not None:
    raw_losses = distillation_losses[response_mask_bool].float()
    clamp_frac = ((raw_losses >  loss_config.loss_max_clamp) |
                  (raw_losses < -loss_config.loss_max_clamp)).float().mean()
    metrics_extra["distillation/clamp_fraction"] = Metric(AggregationType.MEAN, clamp_frac)
metrics.update(metrics_extra)
```

**해석**
- `clamp_fraction` > 1~5% → 가설 (3) saturate. advantage가 ±clamp 두 값만 갖는 binary signal처럼 동작 → entropy gradient noise ↑.
- `logprob_gap_p99` 급등 → 학생-교사 격차가 큰 batch (데이터 분포 변화 또는 학생 발산).

### (C) 분포 집중도 (top-1 / top-5 prob)

평탄화 여부를 entropy 외에 한 번 더 확인하는 직접 측정치. logits 살아있는 자리에서 계산.

**추가 위치**: `verl/workers/engine/fsdp/transformer_impl.py:1140` 근처 (`model_output["entropy"] = entropy` 작성 자리)

```python
# transformer_impl.py:1140 근처, logits가 살아있을 때
with torch.no_grad():
    probs = torch.softmax(logits.float(), dim=-1)
    top1_prob = probs.max(dim=-1).values             # (bsz, seqlen)
    top5_prob = probs.topk(5, dim=-1).values.sum(-1) # (bsz, seqlen)
    model_output["top1_prob"] = top1_prob
    model_output["top5_prob"] = top5_prob
```

그리고 `ppo_loss` 또는 `compute_forward_kl_topk` 안에서 response_mask로 mean/p10/p1 집계.

**해석**
- `top1_prob_p10`이 0.1 아래로 내려가면 **collapse 가설 확정**.
- `top5_prob_mean`이 0.5 미만으로 떨어지면 분포가 매우 평탄.

### (D) Weight / param 통계

가중치 자체가 발산하는지 확인.

**추가 위치**: `verl/workers/actor/dp_actor.py:_optimizer_step()` 호출 직후 (line 678 근처). 매 step은 비용이 비싸므로 N step 주기.

```python
if self._global_step % 50 == 0:
    with torch.no_grad():
        total_sq = 0.0
        max_abs = 0.0
        for p in self.actor_module.parameters():
            if p.requires_grad:
                total_sq += p.detach().float().pow(2).sum().item()
                max_abs = max(max_abs, p.detach().abs().max().item())
        mini_batch_metrics["actor/weight_l2"] = total_sq ** 0.5
        mini_batch_metrics["actor/weight_max_abs"] = max_abs
```

**해석**
- `actor/weight_max_abs`가 튐 → bf16 overflow / optimizer 폭주 의심.
- `actor/weight_l2` 단조 증가 → LR 과대 또는 grad clip 부족.

### (E) Rollout ↔ Train 정책 격차 (engine path 보강)

`rollout_corr_metrics`는 legacy `dp_actor.py:637-644`에 있음. engine path(`verl/workers/utils/losses.py`)에서는 빠져 있을 수 있으므로 동일 함수를 `ppo_loss()` 안에서 호출해 메트릭에 합치는 것을 권장. vllm weight sync가 지연되면 student가 자기가 안 만든 분포에 대해 학습 → entropy 진동.

---

## 4. 메트릭 조합 → 가설 매핑

| 관측 패턴 | 가설 | 후속 조치 |
|---|---|---|
| `entropy_p50` ↑, `top1_prob_p10` ↓, `weight_l2` ↑ | 분포 collapse / weight 발산 | LR ↓, grad clip ↓, optimizer state 점검 |
| `entropy_max`만 ↑, `entropy_p50` 평탄 | outlier 토큰 (EOS / multi-turn 경계 / masking 버그) | `response_mask` 검증, `multiturn_opd_masking_and_loss.md` 확인 |
| `clamp_fraction` ↑, `pg_clipfrac` ↑, `ppo_kl` ↑ | clamp + PPO clip 동시 발동 → 신호 binary화 | `loss_max_clamp` ↑ 또는 LR ↓ |
| `logprob_gap_p99` ↑↑, `abs_loss` ↑ | 학생-교사 격차 큰 batch (데이터 분포 변화) | data shuffle/epoch 경계 확인 |
| `rollout_corr` ↑ | vllm sync 지연/실패 | `update_weights_bucket_megabytes`, `free_cache_engine` 점검, `distillation_vllm_weight_sync_debug.md` 확인 |
| `weight_max_abs` ↑, `grad_norm` ↑ 동시 | bf16 / fused kernel 수치 문제 | `use_fused_kernels=False` 유지, SP 변경 시도 |

---

## 5. 권장 적용 순서 (ROI 순)

세 가지만 먼저 추가해도 대부분 분리 가능:

1. **`losses.py:116-122` — entropy quantile (p50/p90/p99/max)**
   collapse vs outlier 분리.
2. **`losses.py:325-354` — `clamp_fraction` + `logprob_gap_p99`**
   clamp saturate 가설 검증.
3. **`dp_actor.py:_optimizer_step` 직후 — `weight_l2` (50 step 주기)**
   weight 발산 확인.

추가 비용이 큰 (C) top-k prob과 (E) rollout_corr 보강은 위 셋으로 가설을 좁힌 뒤 필요 시 추가.

---

## 관련 문서

- `distillation_tensorboard_metrics.md` — 기존 distillation 메트릭 설명
- `entropy_calculation_slime_vs_verl.md` — entropy 수식 비교
- `distillation_loss_vs_abs_loss_interpretation.md` — clamp 전/후 loss 해석
- `multiturn_opd_masking_and_loss.md` — multi-turn masking
- `distillation_vllm_weight_sync_debug.md` — rollout sync 디버깅
