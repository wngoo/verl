# 멀티 노드 On-Policy KD 학습 가이드 (Ray + NCCL)

**기반 스크립트:** `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`  
**예시 환경:** 2노드 × 8GPU = 총 16 GPU

---

## 목차

1. [분산 학습 아키텍처](#1-분산-학습-아키텍처)
2. [Horovod 사용 불가 이유](#2-horovod-사용-불가-이유)
3. [사전 준비](#3-사전-준비)
4. [Ray 클러스터 구성](#4-ray-클러스터-구성)
5. [스크립트 수정](#5-스크립트-수정)
6. [실행 방법](#6-실행-방법)
7. [Slurm 환경 실행](#7-slurm-환경-실행)
8. [리소스 배치 흐름](#8-리소스-배치-흐름)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 분산 학습 아키텍처

verl은 **Ray를 필수 분산 런타임**으로 사용하는 HybridFlow 설계를 따릅니다.

```
Controller (단일 Python 프로세스, Ray Head 노드에서 실행)
    │
    ├── global_pool  ──── Student FSDP Workers (GPU 0~N)
    │                     Student vLLM Rollout Workers
    │
    └── teacher_pool ──── Teacher vLLM Workers (GPU M~K)
                          (enable_resource_pool=True 시 별도 풀)
```

멀티 노드는 **Ray 클러스터**를 구성하는 것만으로 가능합니다.  
`trainer.nnodes`와 `distillation.teacher_model.nnodes` 값으로 각 풀의 노드 수를 지정합니다.

### 리소스 풀 구성 방식 (`main_ppo.py:227-258`)

```python
# global_pool: Student 전용
resource_pool_spec = {
    "global_pool": [n_gpus_per_node] * trainer.nnodes,
}

# teacher_pool: enable_resource_pool=True 시 생성
resource_pool_spec["teacher_pool"] = [
    distillation.teacher_model.n_gpus_per_node
] * distillation.teacher_model.nnodes
```

> `enable_resource_pool=False`(기본값)이면 Teacher도 global_pool을 공유하며,  
> `distillation.teacher_model.n_gpus_per_node`는 `trainer.n_gpus_per_node`로 **덮어써집니다.**

---

## 2. Horovod 사용 불가 이유

verl 코드베이스 전체에 Horovod 관련 코드가 **전혀 없습니다.** 근본적인 설계 차이가 있습니다.

| 항목 | verl (Ray + NCCL) | Horovod |
|---|---|---|
| 런타임 | Ray (필수) | MPI / Gloo / NCCL AllReduce |
| 실행 모델 | Controller-Worker 비대칭 | 모든 프로세스 동일 스크립트 |
| 리소스 분리 | resource_pool로 Student/Teacher 격리 | 단일 data parallel 전제 |
| Rollout 엔진 | vLLM/SGLang이 Ray Actor로 실행 | 지원 불가 |

Horovod를 추가하려면 `single_controller/ray/` 전체 재구현이 필요하므로 **현실적으로 불가능**합니다.  
멀티 노드 목표라면 아래 Ray 클러스터 방식을 사용하세요.

---

## 3. 사전 준비

### 3.1 모든 노드에 동일 환경 설치

```bash
git clone https://github.com/verl-project/verl.git
cd verl

uv venv --python 3.12
source .venv/bin/activate
pip install -e .[test,vllm]
```

### 3.2 공유 파일시스템 확인

모든 노드에서 **동일한 경로**로 접근 가능해야 합니다 (NFS, GPFS, Lustre 등).

```bash
export DATA_PATH=/shared/data          # 학습 데이터 (parquet)
export MODEL_PATH=/shared/models       # HuggingFace 모델 가중치
export CKPT_PATH=/shared/checkpoints   # 체크포인트 저장 위치
```

> 공유 파일시스템이 없으면 각 노드에 동일 경로로 데이터/모델을 복사해야 합니다.

### 3.3 NCCL 네트워크 환경 변수

```bash
# InfiniBand 환경 (권장, 고속)
export NCCL_IB_HCA=mlx5_0,mlx5_1    # 실제 IB 디바이스명으로 변경
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0         # IB 네트워크 인터페이스
export NCCL_NVLS_ENABLE=1

# Ethernet 환경 (IB 없을 때)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0        # 실제 이더넷 인터페이스명
export NCCL_NVLS_ENABLE=0
```

### 3.4 노드 간 SSH 무암호 접속 확인

```bash
# node0 → node1 무암호 접속 테스트
ssh node1 "hostname"
```

---

## 4. Ray 클러스터 구성

### 4.1 Head 노드 (node0)

```bash
ray start --head \
  --node-ip-address=192.168.1.10 \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=64 \
  --dashboard-host=0.0.0.0

# 성공 시 출력:
# Ray runtime started.
# To connect: ray start --address='192.168.1.10:6379'
# Dashboard: http://192.168.1.10:8265
```

### 4.2 Worker 노드 (node1, node2, ...)

```bash
ray start \
  --address=192.168.1.10:6379 \
  --num-gpus=8 \
  --num-cpus=64
```

### 4.3 클러스터 확인 (node0에서)

```bash
python -c "
import ray
ray.init(address='auto')
resources = ray.cluster_resources()
print('Cluster resources:', resources)
# 정상: {'CPU': 128.0, 'GPU': 16.0, ...}
"
```

### 4.4 클러스터 종료

```bash
# 모든 노드에서
ray stop
```

---

## 5. 스크립트 수정

`run_qwen_gsm8k.sh` 대비 변경 항목만 표시합니다.

### 핵심 변경 사항 요약

| 변수 | 원본 | 멀티 노드 |
|---|---|---|
| `STUDENT_WORLD_SIZE` | 2 | 8 (노드당 GPU 전체) |
| `TEACHER_RESOURCE_POOL` | `False` | `True` (노드 분리 필수) |
| `TEACHER_WORLD_SIZE` | 4 | 8 (노드당 GPU 전체) |
| `TEACHER_NNODES` | (없음) | 1 |
| `STUDENT_NNODES` | (없음) | 1 |
| `TEACHER_TP` | 1 | 4 |
| `TEACHER_NUM_WORKERS` | 8 | 2 (= 8 / 4) |
| `TRAIN_PROMPT_BSZ` | 128 | 256 (데이터 병렬 2배) |
| `ENFORCE_EAGER` | `True` | `False` (성능 우선) |
| `trainer.nnodes` | 1 | 1 (Student 전용) |
| `ray_kwargs.ray_init.num_cpus` | null | 64 (Slurm 필수) |

### 전체 멀티 노드 스크립트

```bash
#!/usr/bin/env bash
set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm"

FAMILY="Qwen"
STUDENT_MODEL=Qwen2.5-0.5B
TEACHER_MODEL=Qwen2.5-3B-Instruct

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_on_policy_distillation_example_gsm8k'

MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=256                   # 노드 2배 → 배치도 2배
STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

# ── 멀티 노드 핵심 설정 (2노드 × 8GPU) ─────────────────
STUDENT_NNODES=1                       # Student 전용 노드 수
STUDENT_WORLD_SIZE=8                   # 노드당 Student GPU 수 (node0 전체)

TEACHER_RESOURCE_POOL=True             # 반드시 True (노드 분리)
TEACHER_NNODES=1                       # Teacher 전용 노드 수
TEACHER_WORLD_SIZE=8                   # 노드당 Teacher GPU 수 (node1 전체)
TEACHER_TP=4                           # vLLM TP (3B 모델, 4GPU per replica)
TEACHER_NUM_WORKERS=2                  # = TEACHER_WORLD_SIZE / TEACHER_TP = 8/4
# ─────────────────────────────────────────────────────────

SP=1
ENFORCE_EAGER=False                    # 멀티 노드: CUDA graph 활성화 (성능 우선)

EXP_NAME="fsdp/multinode-2n8g/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}/pg-${USE_POLICY_GRADIENT}"

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

TRAIN_FILES="['$gsm8k_train_path']"
TEST_FILES="['$gsm8k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${FAMILY}/${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=$TEACHER_NUM_WORKERS
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=$TEACHER_NNODES
    distillation.teacher_model.model_path="${FAMILY}/${TEACHER_MODEL}"
    distillation.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.7
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=$MAX_NUM_TOKENS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=$STUDENT_NNODES
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    trainer.log_val_generations=5
    ray_kwargs.ray_init.num_cpus=64
)

############################ Launch ############################

RAY_ADDRESS=auto python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
```

---

## 6. 실행 방법

Ray 클러스터를 구성한 후, **node0에서만** 스크립트를 실행합니다.  
Ray가 자동으로 node1에 Teacher worker를 배치합니다.

```bash
# node0에서 실행
cd /shared/verl
source .venv/bin/activate

export DATA_PATH=/shared/data
export WANDB_API_KEY=your_key_here

# NCCL 설정
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

bash examples/on_policy_distillation_trainer/run_qwen_gsm8k_multinode.sh
```

### `ray job submit` 방식 (원격 제출)

```bash
ray job submit \
  --address http://192.168.1.10:8265 \
  --working-dir /shared/verl \
  --runtime-env verl/trainer/runtime_env.yaml \
  -- python3 -m verl.trainer.main_ppo \
    --config-path=examples/on_policy_distillation_trainer/config \
    --config-name='ppo_trainer.yaml' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    ...
```

---

## 7. Slurm 환경 실행

클러스터 스케줄러(Slurm)가 있는 환경의 래퍼 스크립트입니다.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=verl-kd-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -xeuo pipefail

source /shared/verl/.venv/bin/activate
export PYTHONPATH=/shared/verl:$PYTHONPATH

# NCCL 설정
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NVLS_ENABLE=1
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1

# 노드 정보 수집
NNODES=$SLURM_JOB_NUM_NODES
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379

echo "Head node: $head_node ($head_node_ip), Total nodes: $NNODES"

# Ray Head 시작 (node0)
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head \
  --node-ip-address="$head_node_ip" \
  --port=$port \
  --num-gpus=8 \
  --num-cpus="${SLURM_CPUS_PER_TASK}" \
  --block &

sleep 15

# Ray Worker 시작 (node1, node2, ...)
for ((i=1; i<NNODES; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray worker at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start \
    --address="$head_node_ip:$port" \
    --num-gpus=8 \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --block &
  sleep 5
done

sleep 15

# 클러스터 확인
python -c "
import ray
ray.init(address='auto')
resources = ray.cluster_resources()
print('Cluster resources:', resources)
expected_gpus = $NNODES * 8
actual_gpus = resources.get('GPU', 0)
assert actual_gpus == expected_gpus, f'GPU mismatch: expected {expected_gpus}, got {actual_gpus}'
print(f'All {int(actual_gpus)} GPUs confirmed across {$NNODES} nodes.')
"

# 학습 실행 (head 노드에서만)
export DATA_PATH=/shared/data
export RAY_ADDRESS=auto

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash /shared/verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_multinode.sh
```

---

## 8. 리소스 배치 흐름

```
Ray Cluster (2노드 × 8GPU = 16GPU)
│
├── global_pool  [node0: GPU 0-7]
│     ├── Student FSDP Workers   (DP=8, 모든 GPU에 샤딩)
│     └── Student vLLM Rollout   (TP=1, 각 GPU 독립 실행)
│
└── teacher_pool [node1: GPU 0-7]
      ├── Teacher vLLM Replica 0  (GPU 0-3, TP=4)
      └── Teacher vLLM Replica 1  (GPU 4-7, TP=4)
```

### `num_workers` 계산 공식

```
num_workers = TEACHER_WORLD_SIZE × TEACHER_NNODES / TEACHER_TP
            = 8 × 1 / 4 = 2
```

`num_workers`는 Teacher vLLM **replica 수**입니다. 높을수록 Teacher 추론 처리량이 증가합니다.

---

## 9. 트러블슈팅

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `ray.init()` 후 GPU가 절반만 보임 | Worker 노드 Ray 연결 실패 | `ray status`로 노드 연결 상태 확인 |
| `NCCL timeout` 또는 학습 hang | 방화벽 또는 잘못된 네트워크 인터페이스 | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` 재설정 |
| Teacher 노드 OOM | `gpu_memory_utilization` 과다 | `0.5~0.6`으로 낮춤 |
| `Module not found` 에러 | 노드 간 코드 경로 불일치 | `PYTHONPATH=/shared/verl` 통일 |
| Slurm에서 Ray hang | `num_cpus=null` (자동 감지 실패) | `ray_kwargs.ray_init.num_cpus=64` 명시 |
| Worker 노드에서 모델 로딩 실패 | 공유 경로 미마운트 | NFS/GPFS 마운트 상태 확인 |
| `enable_resource_pool=False`에서 Teacher GPU 수 무시 | 코드가 trainer 값으로 덮어씀 | 멀티 노드는 반드시 `enable_resource_pool=True` 사용 |

---

## 참고 파일

| 파일 | 설명 |
|---|---|
| `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` | 단일 노드 기준 스크립트 |
| `examples/gspo_trainer/test_gspo_3b_math_slurm.sh` | Slurm 멀티 노드 예시 |
| `examples/sapo_trainer/run_qwen30b_sapo.sh` | 대규모 멀티 노드 예시 |
| `verl/trainer/main_ppo.py:224-263` | 리소스 풀 생성 로직 |
| `verl/trainer/runtime_env.yaml` | Ray runtime 환경 변수 |
| `verl/trainer/constants_ppo.py` | NCCL/VLLM 기본 환경 변수 |
