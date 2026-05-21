# MOPD Teacher Pool Sizing 가이드

Multi-teacher On-Policy Distillation(MOPD)에서 teacher resource pool 크기를 잡을 때
자주 마주치는 에러와 올바른 설정 방법을 정리한다.

---

## 1. Teacher 배치 구조

- MOPD는 학생(student) 풀과 **완전히 분리된 Ray resource pool**(`teacher_pool`)에 teacher를 띄운다.
  - 학생 풀: `trainer.n_gpus_per_node × trainer.nnodes` (= `global_pool`)
  - Teacher 풀: `distillation.n_gpus_per_node × distillation.nnodes` (= `teacher_pool`)
- 두 풀은 Ray placement group 단위로 격리되며, **student–teacher colocate 옵션은 없다.**
- 같은 물리 노드 안에서 GPU만 분리해도 동작하지만(예: 16 GPU 단일 노드에서 8+8), 일반적으로는 노드 단위로 나누는 게 자연스럽다.
- 코드: `verl/trainer/main_ppo.py:241-275`, `verl/workers/config/distillation.py`

---

## 2. 핵심 공식

`verl/workers/config/distillation.py:141-150, 262-269`

```text
per_replica_world_size = inference.tensor_model_parallel_size
                       × inference.data_parallel_size
                       × inference.pipeline_model_parallel_size

teacher.world_size     = num_replicas × per_replica_world_size
```

**검증 조건 (multi-teacher일 때):**

```text
sum(teacher.world_size for teacher in teacher_models)
    ==
distillation.n_gpus_per_node × distillation.nnodes
```

값이 일치하지 않으면 다음과 같은 에러가 발생한다:

```
Sum of teacher (num_replicas * per_replica_world_size) (32) must match
the distillation resource pool size (n_gpus_per_node=8 * nnodes=1 = 8)
```

---

## 3. 자주 빠지는 함정

### 함정 A — `tensor_model_parallel_size` 기본값이 2

`RolloutConfig.tensor_model_parallel_size`의 default는 **2**다 (`verl/workers/config/rollout.py`).
Teacher 설정에서 명시적으로 1로 두지 않으면 silent하게 per_replica가 2배로 부풀어 sum이 어긋난다.

### 함정 B — `num_replicas`를 GPU 수와 혼동

`num_replicas`는 **replica 개수**지 GPU 개수가 아니다.
TP=4면 한 replica가 4 GPU를 차지하므로, "8 GPU에 두 teacher 띄우자"는 의도로
`num_replicas=4, 4`를 주면 sum = `4×4 + 4×4 = 32 ≠ 8` 이 된다.

### 함정 C — pool 크기 합산 공식 오류

스크립트에서 풀 크기를 다음처럼 계산하면 TP를 빠뜨린 것:

```bash
# 잘못됨: num_replicas만 더함
TEACHER_POOL_WORLD_SIZE=$(( TEACHER_NUM_REPLICAS_BASE + TEACHER_NUM_REPLICAS_AGENT ))
```

올바른 합산:

```bash
TEACHER_POOL_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_BASE + TEACHER_NUM_REPLICAS_AGENT) * TEACHER_TP_SIZE ))
```

또한 verl이 받는 `distillation.n_gpus_per_node`는 **per-node** 값이다.
총 teacher GPU가 한 노드 용량을 넘으면 `distillation.nnodes`를 늘리고
`n_gpus_per_node`는 노드당 값으로 나눠 넣어야 한다.

### 함정 D — `teacher_model` 키 silent drop

기본 default 키 `teacher_model`을 multi-teacher에서 다른 이름과 섞어 쓰면
첫 번째 항목이 조용히 사라진다 (`verl/workers/config/distillation.py:222-239`).
Multi-teacher일 때는 항상 `teacher_models.<자유로운_이름>` 으로 명시하라.

---

## 4. 단일 teacher vs multi-teacher

- **단일 teacher**: `num_replicas`를 비워두면 `pool_size // per_replica_world_size`로 자동 계산.
  - 단, `pool_size % per_replica_world_size == 0` 이어야 함.
- **Multi-teacher**: 각 teacher의 `num_replicas`를 **반드시 명시**해야 한다.
  자동 계산이 없고, 합이 풀 크기와 정확히 일치해야 한다.

---

## 5. 실전 예시 — 16 GPU 환경

### 시나리오

- 노드 1 (8 GPU): 학생 학습
- 노드 2 (8 GPU): teacher 풀
  - Teacher 1 (31B, TP=4): 4 GPU
  - Teacher 2 (31B, TP=4): 4 GPU

### 변수 정의

```bash
# 학생
TRAINER_NNODES=1
TRAINER_GPUS_PER_NODE=8

# Teacher 공통
TEACHER_TP_SIZE=4

# 각 teacher 1 replica = TP × 1 = 4 GPU
TEACHER_NUM_REPLICAS_BASE=1
TEACHER_NUM_REPLICAS_AGENT=1

# Teacher 풀 = (1 + 1) × 4 = 8 GPU, 단일 노드
TEACHER_NUM_NODES=1
TEACHER_POOL_WORLD_SIZE=$(( (TEACHER_NUM_REPLICAS_BASE + TEACHER_NUM_REPLICAS_AGENT) * TEACHER_TP_SIZE ))
```

### Hydra 인자

```bash
trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE      # 8
trainer.nnodes=$TRAINER_NNODES                      # 1

distillation.n_gpus_per_node=$TEACHER_POOL_WORLD_SIZE   # 8 (per-node)
distillation.nnodes=$TEACHER_NUM_NODES                  # 1

+distillation.teacher_models.base.num_replicas=$TEACHER_NUM_REPLICAS_BASE
+distillation.teacher_models.base.inference.tensor_model_parallel_size=$TEACHER_TP_SIZE
+distillation.teacher_models.agent.num_replicas=$TEACHER_NUM_REPLICAS_AGENT
+distillation.teacher_models.agent.inference.tensor_model_parallel_size=$TEACHER_TP_SIZE
```

### 검증

```text
per_replica_world_size  = 4 × 1 × 1 = 4
base_teacher.world_size  = 1 × 4 = 4
agent_teacher.world_size = 1 × 4 = 4
Sum                      = 8
pool                     = 8 × 1 = 8
                              ✓
```

---

## 6. 처리량 튜닝 메모

위 설정은 teacher당 replica 1개라 학생 rollout이 매 step teacher 1대에 직렬로 의존한다.
Teacher inference가 병목이면 다음 옵션을 고려:

- **Replica 추가**: teacher pool을 키우고 (`TEACHER_NUM_REPLICAS_*` 증가) 더 많은 replica를 병렬화.
  - 예: 32 GPU teacher 풀에서 base 4 + agent 4 replica → 라우팅 부하 분산.
- **TP 축소**: 모델이 들어간다면 TP를 줄여 replica 밀도를 높임.
  31B 모델은 보통 TP=4 이상이 필요해 적용 어려움.

먼저 위 16 GPU 구성으로 동작을 확인한 뒤 프로파일링 결과에 따라 확장하는 흐름을 권장.

---

## 참고 코드

- 풀 생성·역할 매핑: `verl/trainer/main_ppo.py:241-275`
- 검증 로직 및 `world_size` 공식: `verl/workers/config/distillation.py:141-302`
- Rollout 기본값(`tensor_model_parallel_size=2`): `verl/workers/config/rollout.py`
- Teacher 서버 기동: `verl/experimental/teacher_loop/teacher_model.py`
