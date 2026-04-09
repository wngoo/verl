---
date: 2026-04-09
topic: verl OPD Gemma4 4B(student)+31B(teacher) 128K context 64xH100 TP/PP/메모리 설정
tags: [verl, opd, gemma4, tp, pp, gpu_memory_utilization, 128k, h100, teacher_resource_pool]
---

# verl OPD Gemma4 4B+31B 128K Context 설정 (H100×64)

> 프레임워크: verl On-Policy Distillation (OPD)  
> 하드웨어: H100 80GB × 64 (8 nodes × 8 GPUs)  
> Student: Gemma4 4B, Teacher: Gemma4 31B  
> Context: 128K tokens  
> TEACHER_RESOURCE_POOL=TRUE

---

## 1. 128K Context KV 캐시 메모리 분석

KV 캐시 크기 공식:
```
per_sequence_kv = num_layers × 2(K+V) × num_kv_heads × head_dim × 2bytes × seq_len
```

### Teacher (Gemma4 31B, 추정: 46 layers, 16 KV heads, head_dim=128)
```
per_token  = 2 × 16 × 128 × 2 × 46 = 376 KB/token
128K total = 376 KB × 131,072 ≈ 47 GB (batch_size=1)
→ TP=8로 분산 시 GPU당 ≈ 5.9 GB
```

### Student (Gemma4 4B, 추정: 28 layers, 8 KV heads, head_dim=256)
```
per_token  = 2 × 8 × 256 × 2 × 28 = 229 KB/token
128K total = 229 KB × 131,072 ≈ 29 GB (batch_size=1)
→ TP=4로 분산 시 GPU당 ≈ 7.2 GB
```

---

## 2. GPU 분할 (TEACHER_RESOURCE_POOL=TRUE)

| Pool | GPUs | 노드 | 용도 |
|---|---|---|---|
| Teacher Pool | 32 GPUs | node 0~3 | vLLM 추론 전용 |
| Student Pool | 32 GPUs | node 4~7 | FSDP 학습 + vLLM rollout |

> 128K context에서 teacher 추론이 주 병목이므로 50/50 분할이 안전.

---

## 3. Teacher (31B) 설정

### TP=8, PP=1

```yaml
teacher:
  rollout:
    tensor_model_parallel_size: 8   # 1 노드(8 GPUs) 내 NVLink 활용
    pipeline_model_parallel_size: 1  # 추론은 PP 불필요, latency 최소화
    gpu_memory_utilization: 0.85
    max_model_len: 131072
    enable_chunked_prefill: true
    max_num_batched_tokens: 8192    # 128K prefill 스파이크 방지
```

### 메모리 계산 (TP=8, GPU 1개 기준 80GB)

| 항목 | 크기 | 비고 |
|---|---|---|
| 모델 웨이트 (31B BF16) | 62 GB ÷ 8 = **7.75 GB** | TP=8 분산 |
| KV 캐시 풀 (0.85 적용) | 80 × 0.85 − 7.75 ≈ **60 GB** | 128K sequences |
| per-sequence KV (TP=8) | 47 GB ÷ 8 ≈ **5.9 GB** | 동시 ~10 sequences 가능 |

### Teacher instances
```
32 GPUs ÷ 8 (TP) = 4 teacher instances 동시 운영
```

---

## 4. Student (4B) 설정

### Actor 학습: FSDP (TP=1, PP=1)

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp
    fsdp_config:
      activation_checkpointing: true   # 128K context 필수
```

FSDP는 ZeRO-3 Data Parallelism → TP 개념 없음. 32 GPUs 전체에 분산.

### Rollout (vLLM): TP=2 권장 (colocate 기준)

```yaml
actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    gpu_memory_utilization: 0.50
    max_model_len: 131072
    enable_chunked_prefill: true
    max_num_batched_tokens: 4096
```

### TP 선택별 메모리 비교 (4B, 80GB GPU)

| TP | GPU당 웨이트 | GPU당 KV cache (128K) | GPU당 합계 |
|---|---|---|---|
| 1 | 8 GB | ~29 GB | ~37 GB |
| 2 | 4 GB | ~14.5 GB | ~18.5 GB ✓ |
| 4 | 2 GB | ~7.2 GB | ~9.2 GB (여유 많음) |

---

## 5. gpu_memory_utilization 설정 가이드

### Teacher: 0.85

- 추론 전용이므로 높게 설정 가능
- KV 캐시 풀을 최대한 확보해 128K throughput 향상
- OOM 안전 마진으로 0.90보다 낮게

### Student Rollout: 0.50 (colocate=true 시)

**왜 낮아야 하는가:**

vLLM은 시작 시 `gpu_memory_utilization × GPU_VRAM`을 KV 캐시로 **미리 선점**한다.
Actor 학습(FSDP)과 Rollout(vLLM)이 같은 GPU를 시간 분할로 사용하므로:

```
80 GB × 0.50 = 40 GB (vLLM에 허용)
- 4B 웨이트 (TP=2): 4 GB
- KV 캐시 풀: ~36 GB → 128K 기준 약 1~2 sequences 처리
나머지 40 GB: FSDP 학습 시 gradient, optimizer, activation 용도
```

| 상황 | 권장값 |
|---|---|
| Actor + Rollout 같은 GPU (colocate=true) | **0.45~0.55** |
| Actor + Rollout 분리 (colocate=false) | **0.75~0.85** |

---

## 6. 전체 설정 요약

```yaml
# Teacher Pool (32 GPUs, node 0~3)
teacher:
  rollout:
    tensor_model_parallel_size: 8
    pipeline_model_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_model_len: 131072
    enable_chunked_prefill: true
    max_num_batched_tokens: 8192

# Student Pool (32 GPUs, node 4~7)
actor_rollout_ref:
  actor:
    strategy: fsdp
    fsdp_config:
      activation_checkpointing: true
  rollout:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    gpu_memory_utilization: 0.50
    max_model_len: 131072
    enable_chunked_prefill: true
    max_num_batched_tokens: 4096
```

---

## 7. 주의사항

| 항목 | 내용 |
|---|---|
| **TP 노드 내 유지** | TP는 NVLink 범위(1 노드 8 GPUs) 내에서만 사용. 노드 간 TP는 InfiniBand 대역폭 낭비 |
| **activation_checkpointing** | 128K × 4B 학습 시 checkpointing 없으면 activation만 수백 GB 필요 (필수) |
| **chunked prefill** | `enable_chunked_prefill: true`로 128K prefill 시 GPU 메모리 스파이크 방지 |
| **OOM 발생 시** | student `gpu_memory_utilization`을 0.05씩 낮추거나 rollout TP=4로 올릴 것 |
| **micro_batch_size** | 128K context에서는 `ppo_micro_batch_size_per_gpu=1`이 사실상 강제됨 |

---

## 참고

- 기존 TRL 기반 설정: `claude_docs/trl_onpolicy_kd_gemma4_h100x64.md`
- Resource pool 상세: `claude_docs/verl_distillation_resource_pool_2.md`
- GPU 파라미터 전반: `claude_docs/gpu_settings_guide.md`
