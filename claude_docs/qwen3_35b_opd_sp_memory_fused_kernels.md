# Qwen3 35B On-Policy Distillation: SP / 메모리 / Fused Kernels / K1 Loss 정리

64 H100 환경에서 Qwen3 ~35B 급 모델을 student 로 OPD (on-policy distillation) 돌릴 때 마주치는
- Ulysses SP 사이즈 결정
- backward OOM 시 메모리 줄이는 레버
- `use_fused_kernels` 의 학습 품질 영향
- `loss_mode="k1"` 이 vocab 전체를 쓰는지

네 가지 주제를 정리한 문서.

기준 스크립트: `examples/on_policy_distillation_trainer/run_gemma4_e4b_opd_gemma4_e4b_on.sh`

---

## 1. SP=32 가 적절한가? (64 H100, Qwen3 ~35B)

### verl 의 Ulysses SP 제약 (`verl/models/transformers/monkey_patch.py:338-343`)

```
num_attention_heads % ulysses_sp_size == 0
num_key_value_heads % ulysses_sp_size == 0  OR  ulysses_sp_size % num_key_value_heads == 0
```

Qwen3-32B 기준 (`num_attention_heads=64`, `num_key_value_heads=8`) 으로는 SP=32 도 divisibility 통과 (64%32=0, 32%8=0). 단, KV head 가 8개라 32 rank 에 replicate 되어 메모리 이득은 줄어든다.

### SP=32 가 비효율적인 이유

1. **DP 가 2까지 줄어든다.** 64 GPU / SP=32 = DP=2. `train_batch_size=256` 기준 DP rank 당 128번 grad accumulation → PPO step 한 번이 매우 길어진다.
2. **Cross-node all-to-all 이 layer 마다 발생.** SP=32 면 4 노드 (8 GPU/node) 를 가로지르는 all-to-all 이 transformer layer 마다 forward/backward 에서 일어난다. Ulysses 는 NVLink 대역폭에 민감해 노드 경계를 넘으면 IB 로 떨어지고 step time 의 지배적 요인이 된다.
3. **35B 면 SP 가 그렇게 클 필요가 없다.** FSDP full shard + grad ckpt + offload 조합이면:
   - 모델 state (param+grad+adam): 35B → GPU 당 ~6–7 GB
   - Flash-attn 사용 시 activation 은 `O(seq·hidden)`. 64K context 도 grad ckpt 켜면 layer 당 ~수백 MB 수준
   - 80GB H100 에서 SP=8 이면 충분히 들어간다.

### 권장

- **SP=8 (intra-node) 부터 시작** → NVLink 안에서 끝나서 통신 손실이 가장 적다. DP=8.
- OOM 나면 SP=16 (2노드 span) 으로 → 이때부터 IB 통신 영향 본격적으로 느껴짐.
- SP=32 는 시퀀스가 정말로 길 때 (≥128K급) 만 고려.
- `ppo_micro_batch_size_per_gpu=1`, `ppo_max_token_len_per_gpu=8192` 도 SP 에 맞춰 다시 튜닝 필요.

---

## 2. Backward 에서 OOM — context_length / batch_size 안 건드리고 줄이는 방법

backward 메모리는 거의 항상 **activation memory** 가 원인. micro_batch_size_per_gpu=1, ctx 도 못 줄이면 단위 step 의 activation 양이 고정인데, 이걸 줄이는 레버 4개 — 효과 큰 순서:

### 2.1. `enable_activation_offload=True` (가장 즉효)

`verl/utils/activation_offload.py`, `fsdp_workers.py:649` 에 구현되어 있음. gradient checkpointing 위에 추가로 **저장된 activation 을 CPU 로 내려보내고** backward 직전에 다시 올린다.

```bash
actor_rollout_ref.model.enable_activation_offload=True
```

H100 + PCIe Gen5/NVLink 라 grad ckpt 단독보다 메모리는 더 줄이면서 step time 패널티는 5~15% 수준. param/optimizer offload 와 동일 패턴.

### 2.2. `use_fused_kernels=True` (large vocab 모델에 효과 큼)

Qwen3 35B 는 vocab ~152K. lm_head 출력 logits 이 `seq_len × vocab` 이라 24K+39K=63K 토큰 기준:
- fp32 logits: 63000 × 152000 × 4 bytes ≈ **38 GB** (한 번에 materialize 되면 즉사)
- bf16 라도 19 GB

fused kernel 은 lm_head + cross-entropy + log_prob 을 chunk 단위로 융합 → full logits 를 저장하지 않는다. distillation 에서 K1 loss 처리할 때 가장 큰 메모리 hot spot 이 여기.

```bash
actor_rollout_ref.model.use_fused_kernels=True
```

> 코멘트에 "triton>=3.3 에서 fused_kernel 성능 깨질 수 있음" 경고 (`megatron_actor.py:152`). FSDP path 에서는 보통 문제없음. 실패 시 `fused_kernel_options` 로 chunk size 조정.

### 2.3. `ulysses_sequence_parallel_size` 한 단계 올리기

SP 는 시퀀스 차원 자체를 쪼개서 per-GPU activation 을 **선형으로** 줄인다. SP=8 → SP=16 이면 backward 메모리 절반. OOM 회피 목적이면 SP=16 (2노드 span) 은 허용 범위.

### 2.4. `ppo_max_token_len_per_gpu` 줄이기 (dynamic_bsz 켤 때만)

`USE_DYNAMIC_BSZ=False` 면 거의 무시되지만, dynamic_bsz 활성화하면 micro-batch 의 packed token 수 상한. 동일 train_batch_size 에서 micro 분할만 더 잘게 만드는 효과.

### 추천 적용 순서

1. **`use_fused_kernels=True`** ← 먼저 (대부분 OOM 해결)
2. 그래도 터지면 **`enable_activation_offload=True`** 추가
3. 그래도 부족하면 SP 8→16
4. 마지막 수단으로 `use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu` 낮춤

---

## 3. `use_fused_kernels=True` 가 학습 quality 에 영향 주는가?

**수학적으로 동일한 연산** — 학습 quality 영향 없음.

### 같은 이유

`LinearCrossEntropy` (`verl/utils/kernel/linear_cross_entropy.py`) 는 `lm_head(hidden) → log_softmax → gather(label)` 시퀀스를 chunk 단위로 융합한 것뿐.

| 단계 | False | True |
|---|---|---|
| logits 생성 | `output.logits` 전체 보관 | chunk 별 forward, 저장 안 함 |
| log_prob | `logprobs_from_logits(logits, labels)` | kernel 내부 동일 연산 |
| entropy | `entropy_from_logits(logits)` | kernel 이 함께 반환 |
| temperature | `logits.div_(T)` | kernel 인자 |

`fsdp/transformer_impl.py:1036-1052` — 둘 다 동일한 `log_probs`, `entropy` 텐서를 만들어 동일한 loss path 로. 융합 커널은 fp32 로 내부 누적 (softmax 안정성) → bf16 eager 대비 정밀도 같거나 약간 더 좋음. run-to-run variance 안에 묻히는 수준.

### 호환성 제약 (verl FSDP path)

`use_fused_kernels=True` 일 때 막히는 경우들:

1. **`distillation_use_topk=True` 와 함께 못 씀** (`fsdp/transformer_impl.py:1064-1073`)
   - topk distillation 은 student full logits 에서 top-k 를 뽑아 teacher 와 비교 → fused kernel 은 logits materialize 안 하므로 non-fused 분기에서만 처리.
   - `loss_mode="k1"` 은 이 분기 안 씀 → **안전**.
   - `forward_kl_topk` 로 바꾸려면 fused 못 씀.

2. **per-sample temperature 미지원** (`fsdp/transformer_impl.py:880-882`) — 단일 temperature 면 무관.

3. **`use_prefix_grouper` 와 같이 못 씀** (`dp_actor.py:110-113, 134`) — 안 쓰면 무관.

4. **`calculate_sum_pi_squared` 류 entropy 변형 metric 과 충돌** (`dp_actor.py:110`) — entropy_coeff=0 이면 무관.

### 검증

같은 seed 로 100~200 step 짧게 돌려서 `actor/policy_loss`, `actor/entropy`, distillation loss 가 noise 범위 (보통 < 1%) 안에서 일치하는지 확인.

---

## 4. `loss_mode="k1"` 은 sampled token 의 log_prob 만 사용한다 — vocab 전체 X

코드 (`verl/trainer/distillation/losses.py:325-354`, `core_algos.py:2152-2181`) 확인 결과 정확.

### 동작

```python
# losses.py:341-349
student_log_probs = no_padding_2_padding(model_output["log_probs"], data)  # (bsz, resp_len)
teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
distillation_losses = kl_penalty(
    logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="k1"
)

# core_algos.py:2164-2165
if kl_penalty in ("kl", "k1"):
    return logprob - ref_logprob   # 토큰별 log p_S(y_t) - log p_T(y_t)
```

각 토큰 t 에 대해 **rollout 에서 실제로 샘플된 토큰 y_t 의 log_prob 하나씩만** 사용:
- student: forward 한 번 돌려 `log_probs[t] = log p_S(y_t | x, y_<t)` 추출
- teacher: 미리 계산해 `data["teacher_logprobs"]` 에 저장

shape 이 `(bsz, resp_len)` 인 게 그 증거 — vocab 차원 없음.

### 통계적 의미

Schulman 의 single-sample KL estimator (k1). 토큰을 student 가 샘플 (on-policy) 했을 때:

```
E_{y_t ~ p_S}[ log p_S(y_t) - log p_T(y_t) ] = KL( p_S || p_T )
```

1개 샘플로 **reverse KL** 을 unbiased 추정. 같은 토큰 위치에서 vocab 전체를 보는 `forward_kl_topk` (top-k logit) 나 `full` (전체 vocab, 아직 NotImplementedError) 과 대비.

### 주의: k1 의 gradient 는 unbiased 하지 않음

`core_algos.py:2143-2149` 주석:

> The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3 estimator is not the expected gradient of KL.

값은 unbiased 지만 gradient 는 bias 가 있음. 보정하려면 `loss_mode="k1+"` (`+` suffix) 로 — k2 형태 (`0.5*(logprob-ref_logprob)^2`) 를 straight-through 로 끼워 gradient 는 k2, value 는 k1. 현재 스크립트는 `"k1"` 그대로 → 이 보정 없음.

### Fused kernels 와 K1 의 궁합

K1 이 필요한 입력이 정확히 fused kernel 의 출력 — `log p(y_t)`. full vocab logits 를 절대 만들 일이 없으니 fused_kernel 켜도 정보 손실 0. 반대로 `forward_kl_topk` 는 top-k logit 필요 → full logits 일단 materialize 해야 하므로 fused 못 씀.

---

## 요약 체크리스트 (Qwen3 35B / 64 H100 / OPD k1)

| 설정 | 권장값 | 이유 |
|---|---|---|
| `ulysses_sequence_parallel_size` | **8** (시작), OOM 시 16 | intra-node NVLink, DP=8 확보 |
| `enable_gradient_checkpointing` | True (default) | activation 메모리 절약 |
| `enable_activation_offload` | **True** 추가 권장 | grad ckpt 위에 CPU offload, +5~15% step time |
| `use_fused_kernels` | **True** | vocab ~152K → 19~38GB logits materialize 회피, k1 호환 |
| `fsdp_config.param_offload` | True (이미 켬) | 모델 state CPU offload |
| `fsdp_config.optimizer_offload` | True (이미 켬) | optimizer state CPU offload |
| `ppo_micro_batch_size_per_gpu` | 1 | 더 못 줄임 |
| `loss_mode` | "k1" 또는 "k1+" (gradient 보정) | sampled token only, vocab 전체 X |
