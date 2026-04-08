# gpu_memory_utilization 완전 해설

## 1. 의미: 엔진별로 다르다

`gpu_memory_utilization`의 **의미가 vLLM과 SGLang에서 다르다** (`docs/perf/perf_tuning.rst:35-41` 기준):

| 엔진 | 의미 | 비고 |
|---|---|---|
| **vLLM** (v0.7+) | GPU **전체** 메모리 중 vLLM이 사용할 비율 | 초과분은 사용 안 함 |
| **SGLang** | **여유(free)** 메모리 중 KV cache 등 static 메모리에 할당할 비율 | 나머지 `(1-값)`도 inference 중 추가 사용될 수 있음 |

같은 값이어도 실제 점유 메모리가 다르므로 **엔진을 바꾸면 반드시 재조정** 필요.

기본값: `verl/workers/config/rollout.py:196` → `gpu_memory_utilization: float = 0.5`

---

## 2. 실제 GPU 메모리 사용 구조

GPU 메모리는 크게 두 부분이 경쟁한다:

```
GPU VRAM (예: 80GB A100)
├── [FSDP 학습] 모델 파라미터 + gradient + optimizer state
│     param_offload=False → 파라미터가 GPU에 상주
│     param_offload=True  → CPU로 내려가 GPU 메모리 확보
│
└── [vLLM rollout] 모델 가중치 복사본 + KV cache
      ↑ gpu_memory_utilization이 이 영역을 제어
```

**Colocated 구조** (student rollout이 학습 GPU와 같은 GPU를 공유):

```
80GB
├── FSDP 학습 영역: ~56GB
└── vLLM 영역: 0.3 × 80GB = 24GB
```

값이 높을수록 KV cache 용량이 커져 더 많은 동시 시퀀스 처리 가능.
단, FSDP 학습 영역과 합이 100%를 넘으면 OOM.

---

## 3. CUDA Graph와의 관계

`enforce_eager=False`이면 CUDA Graph가 `gpu_memory_utilization` 한도 **밖에서** 추가 메모리를 사용한다:

```
실제 사용량 = (gpu_memory_utilization × total_vram) + cuda_graph_memory
```

```bash
actor_rollout_ref.rollout.enforce_eager=True   # 메모리 빡빡할 때 (CUDA Graph 비활성화)
actor_rollout_ref.rollout.enforce_eager=False  # 속도 중요, 메모리 여유 있을 때
```

CUDA Graph 캡처 크기를 직접 지정하면 메모리와 성능을 균형 있게 제어 가능:
```bash
actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]  # 메모리 제한 환경
```

---

## 4. 상황별 권장 설정값

### Colocated (학습 + rollout이 같은 GPU)

on-policy KD의 student rollout 또는 GRPO에서 흔한 구조:

```bash
# param_offload=False, optimizer_offload=False
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # ~ 0.4

# param_offload=True, optimizer_offload=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # ~ 0.6
```

### Standalone (rollout 전용 GPU)

teacher 전용 GPU pool 또는 reward model 전용 GPU:

```bash
# enforce_eager=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.7  # ~ 0.8

# enforce_eager=False (CUDA Graph 추가 메모리 고려)
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # ~ 0.7
```

`docs/perf/best_practices.rst:124` 권고: offload 활성화 시 0.8 ~ 0.9도 가능.

### 요약표

| 상황 | 권장값 |
|---|---|
| Colocated, offload 없음 | `0.3 ~ 0.4` |
| Colocated, `param_offload=True` | `0.5 ~ 0.6` |
| Standalone, `enforce_eager=True` | `0.7 ~ 0.8` |
| Standalone, `enforce_eager=False` | `0.6 ~ 0.7` |

---

## 5. On-policy KD 스크립트 설정 전략

`examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` 기준:

```bash
TEACHER_RESOURCE_POOL=False  # teacher가 standalone GPU 4개 단독 사용
STUDENT_WORLD_SIZE=2         # student GPU 2개 (colocated: 학습 + rollout 공유)
TEACHER_WORLD_SIZE=4         # teacher GPU 4개

# 원본 설정
actor_rollout_ref.rollout.gpu_memory_utilization=0.3       # student rollout (colocated)
distillation.teacher_model.inference.gpu_memory_utilization=0.3  # teacher
```

`TEACHER_RESOURCE_POOL=False`여도 teacher는 전용 GPU 4개를 단독으로 사용하므로 teacher 값은 올릴 수 있다:

```bash
# 권장 조정
actor_rollout_ref.rollout.gpu_memory_utilization=0.4       # student: colocated이므로 보수적 유지
                                                            # param_offload=True면 0.5~0.6 가능
distillation.teacher_model.inference.gpu_memory_utilization=0.6  # teacher: 전용 GPU이므로 상향
```

---

## 6. max_num_seqs와의 관계 (OOM 에러 원인)

`gpu_memory_utilization`이 낮아도 `max_num_seqs`가 크면 sampler 워밍업 시 OOM 발생:

```
vLLM 초기화 시: max_num_seqs개의 dummy request로 sampler warmup 수행
→ max_num_seqs가 너무 크면 → "CUDA out of memory occurred when warming up sampler"
```

`run_qwen_gsm8k.sh`의 버그: `MAX_NUM_TOKENS`(토큰 수)를 `max_num_seqs`(시퀀스 수)로 잘못 사용:

```bash
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# ❌ 잘못된 설정 (MAX_PROMPT가 크면 max_num_seqs도 비례해서 커짐)
actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS

# ✅ 올바른 설정 (토큰 수와 시퀀스 수를 분리)
MAX_NUM_SEQS=128
actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS
```

`max_num_seqs` 적정값 계산:
```
가용 KV cache 메모리 = total_vram × gpu_memory_utilization - model_weights
max_num_seqs ≈ 가용 KV cache / (max_seq_len × kv_cache_per_token)

실용적 시작값: 64 ~ 256 (GPU 메모리 보면서 조정)
```

---

## 7. OOM 디버깅 체크리스트

```
OOM 발생 시 체크 순서:

1. max_num_seqs 낮추기 (64 또는 128로 시작)
2. gpu_memory_utilization 낮추기 (0.1씩 감소)
3. enforce_eager=True 설정 (CUDA Graph 메모리 제거)
4. param_offload=True, optimizer_offload=True 설정
5. max_num_batched_tokens 낮추기
6. tensor_model_parallel_size 늘리기 (GPU당 모델 메모리 분산)
```

---

## 참고 코드 위치

| 항목 | 위치 |
|---|---|
| `gpu_memory_utilization` 기본값 정의 | `verl/workers/config/rollout.py:196` |
| vLLM 엔진에 전달 | `verl/workers/rollout/vllm_rollout/vllm_async_server.py:256` |
| SGLang 엔진에 전달 | `verl/workers/rollout/sglang_rollout/async_sglang_server.py:193` |
| 공식 튜닝 가이드 | `docs/perf/perf_tuning.rst` |
| Best practices | `docs/perf/best_practices.rst:123-126` |
