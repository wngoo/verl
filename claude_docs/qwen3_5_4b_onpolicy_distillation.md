# Qwen3.5-4B On-Policy Distillation (OPD) in verl 가능 여부 분석

> 작성일: 2026-04-22  
> 조사 기준: verl `dev-wongoo` 브랜치 (ff7c6a10)

---

## 1. 배경

verl에서 Qwen3.5-4B 모델을 student로 사용하여 on-policy distillation (OPD) 학습이 가능한지 검토한다.  
git commit, PR, 코드베이스를 직접 분석하였다.

---

## 2. Qwen3.5-4B 모델 아키텍처

### 2.1 HuggingFace config.json 기준

| 필드 | 값 |
|---|---|
| `model_type` | `qwen3_5` |
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 16 |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144 (256K context) |
| `tie_word_embeddings` | true |
| `dtype` | bfloat16 |

### 2.2 하이브리드 어텐션 구조

Qwen3.5-4B는 **표준 Transformer가 아닌 하이브리드 어텐션**을 사용한다:

- **28개 linear attention 레이어**: GatedDeltaNet (효율적 선형 어텐션)
- **4개 full attention 레이어**: 표준 self-attention (`full_attention_interval=4`)

```json
"full_attention_interval": 4,
"linear_conv_kernel_dim": 4,
"linear_num_key_heads": 16,
"linear_num_value_heads": 32,
"linear_key_head_dim": 128,
"linear_value_head_dim": 128
```

### 2.3 VLM 구조

Qwen3.5-4B는 **VLM(Vision-Language Model)**이다. `vision_config`를 보유하며 이미지/영상 입력을 처리할 수 있다.  
단, `--language-model-only` 플래그(vLLM) 또는 text-only 데이터로 텍스트 전용 학습도 가능하다.

```
image_token_id: 248056
video_token_id: 248057
vision_config: {24-layer vision encoder, hidden_size=1024}
```

### 2.4 Qwen3 패밀리 혼동 주의

| 모델 패밀리 | model_type | 특성 |
|---|---|---|
| **Qwen3** (예: Qwen3-4B, Qwen3-8B) | `qwen3` | 텍스트 전용 LLM |
| **Qwen3.5** (예: Qwen3.5-4B, Qwen3.5-27B) | `qwen3_5` | VLM (하이브리드 어텐션 포함) |
| **Qwen3-VL** (예: Qwen3-VL-2B) | `qwen3_vl` | VLM (표준 어텐션) |

---

## 3. verl On-Policy Distillation 지원 현황

### 3.1 관련 PR/커밋

#### PR #5041 (커밋 `455e44c6`, 2026-03-21 머지)
```
[fsdp,megatron,vllm,trainer,algo] feat: On-Policy Distillation
```

**지원 기능:**
- FSDP 및 Megatron 백엔드
- top-k distillation loss (forward KL) — `forward_kl_topk`
- KL estimator losses — `k1`, `k3`, `kl`, `abs`, `mse`, `k2`, `low_var_kl`
- Supervised 방식 (`use_policy_gradient=False`)
- Policy-gradient 방식 (`use_policy_gradient=True`)
- vLLM teacher server를 통한 teacher logprobs 계산
- LLM 및 VLM distillation 모두 지원

**추가된 주요 파일:**
```
verl/trainer/distillation/losses.py           # 공통 distillation loss
verl/trainer/distillation/fsdp/losses.py      # FSDP forward_kl_topk 계산
verl/trainer/distillation/megatron/losses.py  # Megatron forward_kl_topk 계산
verl/workers/config/distillation.py           # DistillationConfig 데이터클래스
verl/trainer/config/distillation/distillation.yaml
examples/on_policy_distillation_trainer/      # 예시 스크립트
```

#### PR #5682 (커밋 `4045d670`, 2026-03-30 머지)
```
[fsdp, model] feat: add qwen3.5 fsdp grpo training support.
```

**추가된 내용:**
- `verl/models/transformers/qwen3_5.py`: Qwen3.5 transformer adapter
- `verl/models/transformers/monkey_patch.py`: `qwen3_5`, `qwen3_5_moe` model_type 처리 추가
- Qwen3.5-27B/35B GRPO 예시 스크립트 추가

### 3.2 공식 On-Policy Distillation 예시 스크립트

```
examples/on_policy_distillation_trainer/
├── run_qwen_gsm8k.sh        # Qwen2.5-0.5B(student) ← Qwen2.5-3B-Instruct(teacher)
├── run_qwen3_vl_geo3k.sh    # Qwen3-VL-2B(student) ← Qwen3-VL-4B(teacher)
└── run_qwen_gsmk8_megatron.sh
```

**Qwen3.5-4B student 예시는 없음.**

---

## 4. Qwen3.5-4B 지원 코드 경로 분석

### 4.1 monkey_patch 처리

`verl/models/transformers/monkey_patch.py`:

```python
elif model.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
    from verl.models.transformers.qwen3_5 import forward_with_torch_backend, forward_with_triton_backend
    # fused kernel 패치

elif model.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
    # VLM 패치: Qwen3_5Model, Qwen3_5ForConditionalGeneration 등
    from verl.models.transformers.qwen3_5 import qwen3_5_base_forward, forward_with_normal_backend, ...
    Qwen3_5Model.forward = qwen3_5_base_forward
    Qwen3_5ForConditionalGeneration.forward = forward_with_normal_backend
```

`model_type="qwen3_5"`는 monkey_patch에서 **명시적으로 처리**된다.

### 4.2 텍스트 전용 사용 시 Dummy Vision Forward

`verl/models/transformers/qwen3_5.py`의 `_get_input_embeds`:

```python
if pixel_values is None and pixel_values_videos is None:
    # FSDP: 모든 파라미터가 forward에 참여해야 하므로 dummy 비전 forward 실행
    config = model.config.vision_config
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
    inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()  # 값 기여 없음, gradient만
```

**의미**: 텍스트 데이터만 입력해도 비전 인코더가 항상 실행된다. 메모리/연산 오버헤드는 있으나 FSDP 동작에 필수적이다.

### 4.3 fused kernel 지원

Qwen3.5는 `else` 브랜치에서 `dense_common`이 아닌 전용 `forward_with_torch_backend` / `forward_with_triton_backend`를 사용한다. 단, **하이브리드 어텐션(GatedDeltaNet)에서 fused kernel 호환성은 미확인**이므로 초기 실험에서는 `use_fused_kernels=False` 권장.

---

## 5. 주요 제약사항 및 위험 요소

### 5.1 data_format 주의사항 (forward_kl_topk 한정)

`verl/trainer/distillation/fsdp/losses.py:48`:
```python
# data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5
```

`verl/trainer/distillation/losses.py:192`에도 동일 주석 존재.

**FSDP 엔진 코드 분석** (`transformer_impl.py`):

```python
# use_remove_padding=True 경로 (THD 포맷)
if use_remove_padding:
    ...
    if distillation_use_topk:  # forward_kl_topk 시 True
        outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), ...)

# use_remove_padding=False 경로 (bshd 포맷)
else:
    ...
    # distillation_use_topk 처리 코드 없음!
```

- `forward_kl_topk` + `use_remove_padding=False` 조합: **동작하지 않음**
- `k1`/`k3` estimator 계열: `distillation_use_topk=False`이므로 **data_format 무관하게 동작**
- Qwen3.5-27B GRPO 스크립트는 `use_remove_padding=True`를 사용하므로 이 경로는 일단 작동 가능

### 5.2 버전 의존성 (매우 중요)

`examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp.sh` 첫 줄:
```bash
# dependency: vllm==0.18.0, transformers@<cc7ab9be>
```

**GatedDeltaNet 선형 어텐션**은 특정 버전의 vLLM과 transformers에서만 정상 동작한다.  
버전이 맞지 않으면 vLLM 롤아웃 자체가 실패할 수 있다.

### 5.3 Thinking 모드 토큰

Qwen3.5는 기본적으로 `<think>...</think>` 블록을 생성한다.

- distillation 시 teacher logprobs는 `<think>` 토큰 위치까지 계산됨
- `response_mask`가 thinking 구간을 포함하는지 여부에 따라 loss 계산이 달라짐
- 의도에 따라 thinking 구간을 **mask out**할지 결정 필요

### 5.4 vLLM에서 MoE 모델 패치

`verl/utils/vllm/patch.py`:
```python
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
```

Dense 4B는 MoE가 아니므로 이 패치는 직접 관련 없으나, vLLM 내 Qwen3.5 지원 자체가 `vllm==0.18.0` 기준임을 의미한다.

### 5.5 공식 Distillation 예시 없음

Qwen3.5를 student로 사용하는 on-policy distillation 예시 스크립트가 존재하지 않는다.  
`run_qwen_gsm8k.sh` (Qwen2.5 기반)를 참고해 직접 구성해야 한다.

---

## 6. 권장 설정

### 6.1 Loss 모드 선택

| Loss 모드 | 방식 | Qwen3.5 적합성 | 비고 |
|---|---|---|---|
| `k1` | reverse KL estimator (policy gradient) | **권장** | top-k 불필요, 안정적 |
| `k3` | reverse KL estimator (supervised) | **권장** | top-k 불필요, 논문 검증 |
| `forward_kl_topk` | forward KL (top-k) | **주의** | use_remove_padding=True 필수, 미검증 |

### 6.2 예시 스크립트 구성

`run_qwen_gsm8k.sh` 기반으로 아래 항목을 수정:

```bash
STUDENT_MODEL=Qwen3.5-4B
TEACHER_MODEL=Qwen3.5-7B        # 또는 더 강한 모델

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"     # k3도 가능
USE_FUSED_KERNELS=False          # 초기엔 비활성화 권장

# 모델 설정
actor_rollout_ref.model.path="Qwen/Qwen3.5-4B"
actor_rollout_ref.model.use_remove_padding=True   # Qwen3.5 GRPO 스크립트 기준
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.use_fused_kernels=False

# rollout
actor_rollout_ref.rollout.name=vllm              # vLLM만 검증됨

# distillation
distillation.enabled=True
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.use_task_rewards=False
```

### 6.3 환경 설정

```bash
pip install vllm==0.18.0
# transformers: git checkout cc7ab9be (Qwen3.5 GRPO 스크립트 명시)
```

---

## 7. 결론 요약

| 항목 | 상태 |
|---|---|
| model_type `qwen3_5` verl 지원 | ✅ monkey_patch 처리됨 |
| FSDP GRPO 학습 검증 (27B) | ✅ 공식 스크립트 존재 |
| On-policy distillation 기능 자체 | ✅ PR #5041 머지됨 |
| Qwen3.5-4B distillation 공식 예시 | ❌ 없음 |
| `k1`/`k3` loss 사용 시 동작 가능성 | ✅ (data_format 무관) |
| `forward_kl_topk` + use_remove_padding=True | ⚠️ 이론상 가능, 미검증 |
| vLLM/transformers 버전 의존성 | ⚠️ vllm==0.18.0 필수 |
| Thinking 토큰 처리 | ⚠️ 별도 mask 정책 필요 |
| 텍스트 전용 사용 시 dummy vision forward | ⚠️ 메모리 오버헤드 |

**최종 판단**: 가능하나 공식 검증된 설정이 없으므로, `k1` loss + FSDP + vLLM 조합으로 먼저 소규모 실험 후 확장 권장.
