# MOPD Multi-Teacher 설정 가이드

`run_qwen3_mopd_gsm8k_geo3k.sh`(2-teacher 예시)를 기반으로 multi-teacher
distillation을 구성하는 방법을 정리합니다. 관련 구현은 다음 파일에 있습니다.

- `verl/workers/config/distillation.py` — `DistillationConfig`,
  `DistillationTeacherModelConfig` 정의 및 검증
- `verl/experimental/teacher_loop/teacher_manager.py` — 런타임 라우팅
- `verl/trainer/config/distillation/distillation.yaml` — 기본 config

## 1. 라우팅 동작

- `distillation.teacher_key`(기본 `data_source`)에 지정한 데이터 컬럼 값으로
  teacher가 선택됩니다.
- 각 teacher의 `key`는 데이터 샘플의 `data_source` 값과 정확히 일치해야 합니다.
  일치하지 않으면 `teacher_manager.py`의 `_resolve_teacher_key`에서
  `ValueError`가 발생합니다.
- 단일 teacher일 때는 `key`와 무관하게 모든 샘플이 그 teacher로 라우팅됩니다.

### Sample 단위 처리 흐름

각 sample은 **정확히 하나의 teacher**에게만 보내져 logprob을 계산받습니다
(`agent_loop.py:_compute_teacher_logprobs`).

1. 학생이 batch 전체를 rollout (모든 sample이 student의 response를 받음).
2. 각 sample의 `data_source` 값을 보고 매칭되는 teacher 1개에만 student의
   `prompt + response`를 보내 prompt_logprobs를 추출 (`max_tokens=1`이므로
   teacher는 generation 안 하고 logprob만 반환).
3. sample마다 받은 `teacher_ids`, `teacher_logprobs`가 batch에 합쳐져
   distillation loss 계산에 사용됨.

데이터 분배는 round-robin이 아니라 **라벨 기반**입니다. 한 micro-batch가
거의 다 한 도메인이면 다른 teacher는 그 step에서 idle입니다. 부하 분산은
`num_replicas`로 (자주 등장하는 도메인의 teacher에 replica를 더 많이 할당).

현 구조는 `1 sample → 1 teacher`만 지원합니다. 여러 teacher의 logprob을
mixing(ensemble)하려면 별도 구현이 필요합니다.

## 2. 네이밍 규칙 (중요한 함정)

`distillation.yaml` 기본값에 `teacher_models.teacher_model` 항목이 들어 있습니다.
다른 teacher를 추가하면 이 기본 항목은 자동으로 drop되므로
(`DistillationConfig._resolve_teacher_models`),
**첫 teacher도 반드시 `teacher_model`이 아닌 다른 이름**(예: `gsm8k`,
`teacher_model1`)으로 `+`를 붙여 등록해야 합니다.

```bash
# BAD — teacher_model 항목이 silently drop되어 teacher_model2만 사용됨
distillation.teacher_models.teacher_model.key=openai/gsm8k
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k

# GOOD — 첫 항목도 다른 이름으로
+distillation.teacher_models.teacher_model1.key=openai/gsm8k
+distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
```

## 3. GPU pool 크기 계산

`DistillationConfig.__post_init__`에서 다음을 검증합니다.

```
n_gpus_per_node * nnodes == Σ_teachers (num_replicas * per_replica_world_size)
per_replica_world_size = inference.tensor_model_parallel_size
                        * inference.data_parallel_size
                        * inference.pipeline_model_parallel_size
```

합계가 맞지 않으면 즉시 `ValueError`가 발생합니다. 큰 teacher에 `TP>1`을 쓰면
해당 teacher의 `per_replica_world_size`가 늘어나니 합계를 다시 계산해야 합니다.

## 4. 3개 teacher 추가 예시 (gsm8k + geo3k + math)

```bash
ROLLOUT_NAME="vllm"
ENFORCE_EAGER=False
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

TEACHER_NUM_REPLICAS_GSM8K=1
TEACHER_NUM_REPLICAS_GEO3K=1
TEACHER_NUM_REPLICAS_MATH=2     # 예: TP=1이면 GPU 2개 사용
TEACHER_POOL_WORLD_SIZE=$(( TEACHER_NUM_REPLICAS_GSM8K \
                          + TEACHER_NUM_REPLICAS_GEO3K \
                          + TEACHER_NUM_REPLICAS_MATH ))   # = 4

DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=$TEACHER_POOL_WORLD_SIZE
    distillation.nnodes=1

    # --- gsm8k teacher ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="Qwen/Qwen3-4B-Instruct-2507"
    +distillation.teacher_models.gsm8k.num_replicas=$TEACHER_NUM_REPLICAS_GSM8K
    +distillation.teacher_models.gsm8k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.8
    +distillation.teacher_models.gsm8k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.enforce_eager=$ENFORCE_EAGER

    # --- geo3k teacher (VL) ---
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
    +distillation.teacher_models.geo3k.model_path="Qwen/Qwen3-VL-4B-Instruct"
    +distillation.teacher_models.geo3k.num_replicas=$TEACHER_NUM_REPLICAS_GEO3K
    +distillation.teacher_models.geo3k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.8
    +distillation.teacher_models.geo3k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.enforce_eager=$ENFORCE_EAGER
    +distillation.teacher_models.geo3k.inference.engine_kwargs.vllm.mm_processor_cache_gb=0

    # --- math teacher (새로 추가, replica 2개) ---
    +distillation.teacher_models.math.key="DigitalLearningGmbH/MATH-lighteval"
    +distillation.teacher_models.math.model_path="Qwen/Qwen2.5-Math-7B-Instruct"
    +distillation.teacher_models.math.num_replicas=$TEACHER_NUM_REPLICAS_MATH
    +distillation.teacher_models.math.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.math.inference.gpu_memory_utilization=0.8
    +distillation.teacher_models.math.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.math.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.math.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.math.inference.enforce_eager=$ENFORCE_EAGER

    # --- loss 설정은 공통 ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

TRAIN_FILES="['$gsm8k_train_path','$geo3k_train_path','$math_train_path']"
TEST_FILES="['$gsm8k_test_path','$geo3k_test_path','$math_test_path']"
```

## 5. 데이터 라우팅: `data_source` 채워넣기

`data_source`는 **데이터 row마다 본인이 직접 채워넣어야 하는 컬럼**입니다.
teacher 이름을 넣어도 되고 huggingface 데이터셋 ID를 넣어도 됩니다 — 라우팅
관점에서는 **teacher의 `key`와 정확히 같은 문자열**이기만 하면 임의 문자열
모두 OK입니다.

### 매칭 규칙

```
row["data_source"]  ==  distillation.teacher_models.<entry>.key
```

YAML entry 이름(`gsm8k`, `geo3k`, `math` 등)은 라우팅에 쓰이지 않습니다.
`DistillationConfig._resolve_teacher_models`에서 dict가 `key` 필드값으로
다시 매핑되므로 **`key` 필드값**만 비교됩니다.

| 데이터 row의 `data_source` | teacher 설정의 `key`             | 매칭 |
| -------------------------- | -------------------------------- | ---- |
| `"openai/gsm8k"`           | `gsm8k.key="openai/gsm8k"`       | ✓    |
| `"hiyouga/geometry3k"`     | `geo3k.key="hiyouga/geometry3k"` | ✓    |
| `"my_math_teacher"`        | `math.key="my_math_teacher"`     | ✓    |

### 전처리 단계에서 `data_source` 추가

`examples/data_preprocess/gsm8k.py`처럼 row dict에 직접 박아넣고 parquet으로
저장합니다.

```python
data = {
    "data_source": "openai/gsm8k",   # 라우팅 키
    "prompt": [{"role": "user", "content": question}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": solution},
    "extra_info": {...},
}
```

커스텀 데이터를 합쳐 학습할 때:

```python
dataset_a = dataset_a.map(lambda ex: {**ex, "data_source": "my_math"})
dataset_b = dataset_b.map(lambda ex: {**ex, "data_source": "my_vision"})
combined = datasets.concatenate_datasets([dataset_a, dataset_b])
combined.to_parquet("train.parquet")
```

학습 스크립트:

```bash
+distillation.teacher_models.t1.key="my_math"
+distillation.teacher_models.t1.model_path="..."
+distillation.teacher_models.t2.key="my_vision"
+distillation.teacher_models.t2.model_path="..."
```

### `data_source`는 reward 라우팅과 공유됨

verl reward manager도 보통 `data_source`로 reward function을 라우팅합니다.
같은 도메인이 reward와 teacher 양쪽을 결정하는 게 자연스러우면 그대로 두고,
분리가 필요하면 다른 컬럼으로 바꿀 수 있습니다.

```bash
distillation.teacher_key=teacher_id
```

이때 row에 별도의 `teacher_id` 컬럼을 추가해두면 reward 라우팅과 분리됩니다.

### 디버깅 팁

- row의 `data_source` 값에 매칭되는 teacher가 없으면 즉시
  `ValueError: No teacher configured for routing key '...'`로 죽습니다
  (`teacher_manager.py:_resolve_teacher_key`). 메시지에 configured key 목록이
  같이 출력되므로 오타 잡기 쉽습니다.
- parquet 생성 후 한 번 확인:
  ```python
  import pyarrow.parquet as pq
  pq.read_table("train.parquet").to_pandas()["data_source"].unique()
  ```
  unique 값이 의도한 teacher key 집합과 같은지 확인하세요.

## 6. Teacher GPU 분할 레이아웃

여러 teacher를 각자 다른 GPU 묶음에 **따로 띄울 수 있습니다**. 예를 들어
8 GPU pool에서 4 teacher × 2 GPU each는 표준 케이스입니다.

### 동작 원리

`MultiTeacherModelManager._initialize_teacher_model_managers`
(`verl/experimental/teacher_loop/teacher_model.py`)가 통합 teacher pool을
각 teacher의 `world_size`만큼 잘라 분리해 띄웁니다.

```python
split_sizes = [teacher.world_size for teacher in teacher_models.values()]
split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)
# 각 teacher는 자기 split_pool에서 독립된 vLLM/SGLang 서버로 기동됨
```

각 teacher는 별도 inference server이고, 할당된 GPU는 그 teacher 전용입니다.

### 검증 식 (재확인)

```
n_gpus_per_node * nnodes  ==  Σ (num_replicas × TP × DP × PP)
```

8 GPU = 4 teacher × `num_replicas=1` × `TP=2` × `DP=1` × `PP=1` ✓

### Config 예시 (4 teacher × 2 GPU each, 1 node)

```bash
TEACHER_POOL_WORLD_SIZE=8

DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=$TEACHER_POOL_WORLD_SIZE
    distillation.nnodes=1

    # --- teacher 1 (2 GPU) ---
    +distillation.teacher_models.t1.key="domain_a"
    +distillation.teacher_models.t1.model_path="..."
    +distillation.teacher_models.t1.num_replicas=1
    +distillation.teacher_models.t1.inference.name=vllm
    +distillation.teacher_models.t1.inference.tensor_model_parallel_size=2
    +distillation.teacher_models.t1.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t1.inference.max_model_len=$MAX_NUM_TOKENS

    # --- teacher 2 (2 GPU) ---
    +distillation.teacher_models.t2.key="domain_b"
    +distillation.teacher_models.t2.model_path="..."
    +distillation.teacher_models.t2.num_replicas=1
    +distillation.teacher_models.t2.inference.name=vllm
    +distillation.teacher_models.t2.inference.tensor_model_parallel_size=2
    +distillation.teacher_models.t2.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t2.inference.max_model_len=$MAX_NUM_TOKENS

    # --- teacher 3, 4도 동일 패턴 ---
)
```

### "2 GPU"에 들어가는 두 가지 방법

GPU 2개를 어떻게 쓰느냐는 자유입니다.

| 방식                | 설정                       | per_replica_world_size | 의미                                |
| ------------------- | -------------------------- | ---------------------- | ----------------------------------- |
| TP=2                | `TP=2, num_replicas=1`     | 2                      | 한 모델을 2 GPU에 텐서 병렬로 분산  |
| 1 GPU × 2 replicas  | `TP=1, num_replicas=2`     | 1 (×2 replica)         | 같은 teacher 2벌을 띄워 부하 분산   |

큰 모델(7B+)은 TP=2가 일반적이고, 작은 모델인데 라우팅 부하가 큰 도메인은
replica 2개 쪽이 보통 더 효율적입니다.

### 비대칭 조합도 가능

검증 식만 만족하면 자유롭게 조합할 수 있습니다. 예를 들어 8 GPU에서:

- 큰 teacher 1개(TP=4) + 작은 teacher 2개(TP=1×1 replica) + 중간 teacher 1개(TP=2)  
  → 4 + 1 + 1 + 2 = 8 ✓
- 자주 등장하는 도메인의 teacher만 `num_replicas=2` (TP=1 × 2 replica = 2 GPU)

### 주의

- 각 teacher가 별도 vLLM 서버라 모델 가중치가 각자 GPU 메모리에 따로
  적재됩니다. 4 × 7B teacher면 GPU 메모리 부담이 커지므로
  `gpu_memory_utilization`을 0.5~0.7로 낮추는 편이 안전합니다
  (기본 0.8은 단독 teacher 기준).
- TP>1인 teacher는 그 GPU들이 NVLink로 잘 연결돼 있어야 throughput이
  유지됩니다. 같은 node 내 NUMA/NVLink 도메인에 묶이도록 cluster placement를
  확인하세요.
- 학생 GPU는 별개 (`trainer.n_gpus_per_node`). 위 teacher pool 8 GPU는
  teacher 전용이며, 학생은 다른 GPU를 써야 합니다.

## 7. 체크리스트

- 데이터셋의 `data_source` 컬럼 값이 각 teacher의 `key`와 1:1로 매핑되는지
  (parquet 생성 시 지정).
- 큰 teacher에 `TP>1`을 쓰면 해당 teacher의 `per_replica_world_size`가 변하므로
  GPU 합계 재계산 필요.
- VL teacher는 `inference.engine_kwargs.vllm.mm_processor_cache_gb=0` 추가.
- top-k 손실(`forward_kl_topk` 등)을 쓰는 경우
  `inference.engine_kwargs.vllm.max_logprobs >= distillation_loss.topk`가
  성립해야 합니다. 자동 보정되지만 명시 권장
  (`DistillationTeacherModelConfig._validate_topk_logprobs`).
- 현재 `temperature=1.0`만 지원됩니다 (`_get_teacher_sampling_params`).

## 8. 5 teacher × 4 GPU + student 44 GPU (총 64 GPU) 예시

5개 teacher에 각 4 GPU씩(=20 GPU) 할당하고 학생에 44 GPU를 주는 시나리오.
`run_qwen_gsm8k.sh` 기반 변경점만 정리.

### 8.1 검증 식 먼저 맞추기

`DistillationConfig.__post_init__`이 다음을 강제 검증:

```
distillation.n_gpus_per_node * distillation.nnodes
    == Σ_teachers (num_replicas × TP × DP × PP)
```

- Teacher pool: 5 × 4 = **20 GPU** → 위 식 충족
- Student pool: **44 GPU** → 검증 식과 무관, `trainer.n_gpus_per_node *
  trainer.nnodes = 44`만 만족하면 됨
- 두 pool은 독립된 Ray resource pool
  (`MultiTeacherModelManager._initialize_teacher_model_managers`가 teacher pool만
  자르고, 학생은 `trainer.*`로 별도 placement)

### 8.2 Teacher 1대당 4 GPU 쓰는 세 경로

`per_replica_world_size = TP × DP × PP`. 4 GPU를 쓰는 옵션 (모두 검증 식 만족):

| 옵션 | 설정                       | 적합한 상황                          |
| ---- | -------------------------- | ------------------------------------ |
| A    | `TP=4, replicas=1`         | 7B+ 큰 teacher (메모리 안 들어감)    |
| B    | `TP=2, replicas=2`         | 4B 정도 + 그 도메인 트래픽 큼        |
| C    | `TP=1, replicas=4`         | 1~3B 작은 teacher + 라우팅 부하 큼   |

5 teacher가 모두 같은 옵션일 필요 없음. 합계 20만 맞으면 비대칭 OK.

### 8.3 Quick Config 변경

```bash
ROLLOUT_NAME="vllm"
ENFORCE_EAGER=False
USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=True

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_mopd_5teachers'

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=2048
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=512
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

# === Student: 44 GPU ===
# 8GPU/node 클러스터에서 44는 단일 균등 PG로 표현 불가. 4GPU/node × 11노드,
# 또는 한 노드를 4GPU 단위로 쪼개 11개 슬롯으로 등록해야 함.
STUDENT_NNODES=11
STUDENT_GPUS_PER_NODE=4              # 11 × 4 = 44 ✓
STUDENT_WORLD_SIZE=$(( STUDENT_NNODES * STUDENT_GPUS_PER_NODE ))

# === Teacher pool: 20 GPU = 5 teacher × 4 GPU each ===
TEACHER_POOL_PER_NODE=4
TEACHER_NNODES=5
TEACHER_POOL_WORLD_SIZE=$(( TEACHER_POOL_PER_NODE * TEACHER_NNODES ))   # 20

SP=4   # 1B+ student × prompt 1k+/response 2k면 ulysses SP 4 권장
```

### 8.4 DISTILLATION 블록 (5 teacher 등록)

§2의 함정 — **첫 teacher도 절대 `teacher_model` 이름 쓰지 말 것**(기본값과 충돌해
silently drop). 모두 `+`로 새 키로 등록.

```bash
DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=$TEACHER_POOL_PER_NODE   # 4
    distillation.nnodes=$TEACHER_NNODES                   # 5

    # --- t1: gsm8k (Qwen2.5-Math-7B → TP=4, 옵션 A) ---
    +distillation.teacher_models.t1.key="openai/gsm8k"
    +distillation.teacher_models.t1.model_path="Qwen/Qwen2.5-Math-7B-Instruct"
    +distillation.teacher_models.t1.num_replicas=1
    +distillation.teacher_models.t1.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.t1.inference.tensor_model_parallel_size=4
    +distillation.teacher_models.t1.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t1.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.t1.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.t1.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.t1.inference.enforce_eager=$ENFORCE_EAGER

    # --- t2: geo3k (Qwen3-VL-8B → TP=4, VL이라 mm cache off) ---
    +distillation.teacher_models.t2.key="hiyouga/geometry3k"
    +distillation.teacher_models.t2.model_path="Qwen/Qwen3-VL-8B-Instruct"
    +distillation.teacher_models.t2.num_replicas=1
    +distillation.teacher_models.t2.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.t2.inference.tensor_model_parallel_size=4
    +distillation.teacher_models.t2.inference.gpu_memory_utilization=0.6
    +distillation.teacher_models.t2.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.t2.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.t2.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.t2.inference.enforce_eager=$ENFORCE_EAGER
    +distillation.teacher_models.t2.inference.engine_kwargs.vllm.disable_mm_preprocessor_cache=True

    # --- t3: math (Qwen2.5-Math-7B → TP=4) ---
    +distillation.teacher_models.t3.key="DigitalLearningGmbH/MATH-lighteval"
    +distillation.teacher_models.t3.model_path="Qwen/Qwen2.5-Math-7B-Instruct"
    +distillation.teacher_models.t3.num_replicas=1
    +distillation.teacher_models.t3.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.t3.inference.tensor_model_parallel_size=4
    +distillation.teacher_models.t3.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t3.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.t3.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.t3.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.t3.inference.enforce_eager=$ENFORCE_EAGER

    # --- t4: code (Qwen2.5-Coder-7B → TP=4) ---
    +distillation.teacher_models.t4.key="bigcode/humanevalpack"
    +distillation.teacher_models.t4.model_path="Qwen/Qwen2.5-Coder-7B-Instruct"
    +distillation.teacher_models.t4.num_replicas=1
    +distillation.teacher_models.t4.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.t4.inference.tensor_model_parallel_size=4
    +distillation.teacher_models.t4.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t4.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.t4.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.t4.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.t4.inference.enforce_eager=$ENFORCE_EAGER

    # --- t5: general (1.5B → TP=1 × replicas=4, 옵션 C로 부하분산) ---
    +distillation.teacher_models.t5.key="my_general"
    +distillation.teacher_models.t5.model_path="Qwen/Qwen2.5-1.5B-Instruct"
    +distillation.teacher_models.t5.num_replicas=4
    +distillation.teacher_models.t5.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.t5.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.t5.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.t5.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.t5.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.t5.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.t5.inference.enforce_eager=$ENFORCE_EAGER

    # --- loss 공통 ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)
```

검증 식 재확인:
`t1(1×4) + t2(1×4) + t3(1×4) + t4(1×4) + t5(4×1) = 4+4+4+4+4 = 20 == 4×5` ✓

### 8.5 TRAINER 블록 (학생 44 GPU)

```bash
TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_GPUS_PER_NODE   # 4
    trainer.nnodes=$STUDENT_NNODES                   # 11
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)
```

> **주의**: 8GPU/노드 클러스터에서 44는 단일 균등 PG로 표현 불가. 4GPU/노드 ×
> 11노드로 잡거나, 한 노드를 4GPU 단위로 쪼개 11슬롯으로 등록해야 함. 8GPU/노드
> 8노드(=64) 클러스터라면 학생 40 + teacher 24처럼 **8의 배수**로 재할당하는
> 게 실무적으로 더 깔끔함.

### 8.6 데이터 라우팅

각 학습 row의 `data_source` 값이 위 5개 `key` 중 하나와 정확히 일치해야 함
(불일치 시 `teacher_manager._resolve_teacher_key`에서 즉시 `ValueError`).

```bash
TRAIN_FILES="['$gsm8k_train','$geo3k_train','$math_train','$code_train','$general_train']"
TEST_FILES ="['$gsm8k_test','$geo3k_test','$math_test','$code_test','$general_test']"
```

각 parquet 만들 때 `data_source`를 t1.key~t5.key와 동일 문자열로 박아둘 것.

### 8.7 흔한 실수 체크리스트

- **`teacher_model`이라는 이름 금지**. `_resolve_teacher_models`가 기본값 항목을
  silently drop함 (§2). 위에선 `t1~t5`로 회피.
- **VL teacher (`t2`)**: `engine_kwargs.vllm.disable_mm_preprocessor_cache=True`
  필수.
- **`forward_kl_topk`** 등 top-k loss 사용 시
  `inference.engine_kwargs.vllm.max_logprobs >= topk`. 자동 보정되지만 명시 권장.
- **`gpu_memory_utilization`**: 단일 teacher용 기본 0.8은 multi-teacher에서
  위험. 각 teacher가 별도 vLLM 서버라 메모리 고립됨 → 0.6~0.7로.
- **temperature=1.0만 지원** (`_get_teacher_sampling_params`).
- **`teacher_key`**: `data_source`는 reward 라우팅과 공유. 분리 필요하면
  `distillation.teacher_key=teacher_id`로 바꾸고 row에 `teacher_id` 컬럼 추가.

요약: ① teacher pool 검증식(`n_gpus_per_node × nnodes = Σ replicas×TP`)을
**20**으로 맞추고 ② 학생 `trainer.n_gpus_per_node × trainer.nnodes = 44`로 별도
설정 ③ teacher 등록 키는 `teacher_model` 금지 — 이 세 가지가 핵심.
