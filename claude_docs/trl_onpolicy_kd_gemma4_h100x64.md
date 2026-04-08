# TRL On-Policy KD: Gemma4 4B(Student) + 31B(Teacher) on H100×64

> 학습 프레임워크: TRL (GKDTrainer)  
> 하드웨어: H100 80GB × 64 (8 nodes × 8 GPUs)  
> 모델: Dense (non-MoE)

---

## 목차

1. [개요 및 On-Policy KD 흐름](#1-개요-및-on-policy-kd-흐름)
2. [메모리 분석](#2-메모리-분석)
3. [GPU 배분 전략](#3-gpu-배분-전략)
4. [Teacher vLLM 설정](#4-teacher-vllm-설정)
5. [Student TRL 학습 설정](#5-student-trl-학습-설정)
6. [DeepSpeed ZeRO-3 상세 설정](#6-deepspeed-zero-3-상세-설정)
7. [Accelerate 설정](#7-accelerate-설정)
8. [실행 스크립트](#8-실행-스크립트)
9. [주요 주의사항](#9-주요-주의사항)

---

## 1. 개요 및 On-Policy KD 흐름

**On-Policy KD**는 student 모델이 자신의 현재 분포로 시퀀스를 생성하고, teacher의 log probability로 KL divergence를 최소화하는 방식이다.

```
[매 스텝]
  1. Student가 prompt에 대해 응답 생성 (vLLM 또는 HF generate)
  2. Teacher가 해당 응답에 대한 token-level log prob 계산
  3. KL(student || teacher) 최소화
  4. Student 가중치 업데이트
```

TRL의 `GKDTrainer`는 이 흐름을 구현하며 핵심 파라미터는:
- `lmbda`: KD loss와 SFT loss의 혼합 비율 (0=순수 KD, 1=순수 SFT)
- `beta`: KL divergence 가중치
- `temperature`: 소프트 레이블 온도

---

## 2. 메모리 분석

| 모델 | 파라미터 | BF16 가중치 | Adam 옵티마이저(FP32) | 합계 |
|------|---------|------------|----------------------|------|
| Student 4B (dense) | ~4B | ~8 GB | ~32 GB | ~40 GB |
| Teacher 31B (dense) | ~31B | ~62 GB | 추론 전용 | ~62 GB |

### Teacher vLLM 메모리 (TP=2 기준, GPU 1개당)
```
가중치:  62 GB / 2 = 15.5 GB
KV cache: 80 GB × 0.88 - 15.5 ≈ 55 GB
→ H100 80GB에 안정적으로 적재 가능
```

### Student 학습 메모리 (ZeRO-3, 32 GPU 기준)
```
가중치/GPU:    8 GB / 32 = 0.25 GB
그래디언트/GPU: 8 GB / 32 = 0.25 GB
옵티마이저/GPU: 32 GB / 32 = 1.0 GB
스테디-스테이트: ~1.5 GB/GPU
활성화(배치=2): ~4-8 GB/GPU
→ 총 ~10 GB/GPU 수준 (여유 충분)
```

---

## 3. GPU 배분 전략

### 배치도 (8 nodes × 8 GPUs)

```
┌─────────────────────────────────────────────────────────┐
│  Node 0-3  (32 GPUs) : Teacher vLLM 서버                │
│    - 31B dense, TP=2                                     │
│    - 16 replicas × 2 GPUs = 32 GPUs                     │
│    - 포트 8000~8015                                      │
│                                                          │
│  Node 4-7  (32 GPUs) : Student TRL 학습 클러스터         │
│    - 4B dense, DP=32, DeepSpeed ZeRO-3                  │
│    - Global batch size = 2 × 4 × 32 = 256               │
└─────────────────────────────────────────────────────────┘
```

### 병렬화 요약

| 컴포넌트 | TP | PP | EP | DP | GPUs |
|---------|----|----|----|----|------|
| Teacher vLLM (31B) | **2** | 1 | 없음 | 16 replicas | 32 |
| Student Train (4B) | 1 | 1 | 없음 | **32** (ZeRO-3) | 32 |

> **TP=2 선택 이유**: `num_attention_heads=16` → TP=2 시 8 heads/GPU ✓  
> **EP 없음**: Dense 모델이므로 Expert Parallelism 불필요  
> **DP=32**: `256 / 32 = 8` (per_device × grad_accum = 2 × 4 = 8) ✓

### Batch size 계산 검증

```
global_batch = per_device_batch × grad_accum × num_gpus
            = 2 × 4 × 32
            = 256 ✓

※ DP=48은 256/48=5.33 → 정수 아님, 사용 불가
   256은 2^8이므로 DP는 반드시 2의 거듭제곱 사용
```

---

## 4. Teacher vLLM 설정

### Python API

```python
from vllm import LLM, SamplingParams

teacher_llm = LLM(
    model="google/gemma-4-31b-it",
    dtype="bfloat16",

    # ─── 병렬화 (dense 모델) ──────────────────────────
    tensor_parallel_size=2,        # 31B → 15.5GB/GPU, heads=16 → 8/GPU
    pipeline_parallel_size=1,      # 추론에서 PP는 bubble로 비효율

    # ─── 메모리 ───────────────────────────────────────
    gpu_memory_utilization=0.88,   # ~55GB KV cache 확보
    max_model_len=4096,
    max_num_seqs=128,              # 동시 시퀀스 수

    # ─── 처리량 최적화 ────────────────────────────────
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    enforce_eager=False,           # CUDA graph 활성화
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)
```

### 서버 기동 스크립트 (16 replicas, Node 0-3)

```bash
#!/bin/bash
# launch_teacher_vllm.sh
# Node 0-3의 모든 GPU(0~31)에서 실행
# 각 노드에서 노드 내 GPU 0-7을 담당

NODE_RANK=${1:-0}   # 0, 1, 2, 3 중 해당 노드 번호

for i in 0 1 2 3; do
  GPU_A=$(( i * 2 ))
  GPU_B=$(( i * 2 + 1 ))
  REPLICA_ID=$(( NODE_RANK * 4 + i ))

  CUDA_VISIBLE_DEVICES=$GPU_A,$GPU_B \
  python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31b-it \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --port $((8000 + REPLICA_ID)) \
    --host 0.0.0.0 &
done

wait
```

---

## 5. Student TRL 학습 설정

### GKDConfig

```python
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

training_args = GKDConfig(
    output_dir="./gemma4-4b-kd-output",

    # ─── 배치 크기 ────────────────────────────────────
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # 2 × 4 × 32 = 256 ✓

    # ─── On-policy KD 핵심 파라미터 ───────────────────
    temperature=0.9,       # 소프트 레이블 온도 (권장: 0.7~1.5)
    lmbda=0.5,             # KD/SFT 혼합: 0.5 = 50% KD + 50% SFT
    beta=1.0,              # KL divergence 가중치

    # ─── 생성 설정 ────────────────────────────────────
    max_new_tokens=512,    # student 생성 최대 길이
    max_length=1024,       # prompt + response 최대 길이

    # ─── 옵티마이저 ───────────────────────────────────
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    weight_decay=0.01,
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,

    # ─── H100 최적화 ──────────────────────────────────
    bf16=True,
    tf32=True,               # H100 TF32 활성화 (약 2× 속도)
    gradient_checkpointing=True,
    optim="adamw_torch_fused",

    # ─── DeepSpeed ────────────────────────────────────
    deepspeed="deepspeed_zero3.json",

    # ─── 로깅/저장 ────────────────────────────────────
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    eval_steps=100,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    report_to="wandb",
)
```

### Trainer 초기화

```python
# 학생 모델 (BF16, 학습 대상)
student_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-4b-it",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 교사 모델 옵션 선택
# ─── 옵션 A: 4-bit 양자화로 같은 프로세스에 로드 ─────
# (ZeRO-3와 충돌 주의 → 교사는 device_map="auto"로 별도 관리)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
teacher_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31b-it",
    quantization_config=bnb_config,
    device_map="auto",
)
# 교사는 학습 불필요
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# ─── 옵션 B: 별도 vLLM 서버 사용 (권장) ──────────────
# → teacher_model 파라미터 대신 vllm_server_host 사용
# training_args.vllm_server_host = "node0"
# training_args.vllm_server_port = 8000

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-4b-it")

trainer = GKDTrainer(
    model=student_model,
    teacher_model=teacher_model,   # 옵션 A
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

## 6. DeepSpeed ZeRO-3 상세 설정

### ZeRO 단계 비교

| 단계 | 분산 대상 | 메모리 절약 | 통신 오버헤드 |
|------|---------|------------|-------------|
| ZeRO-1 | 옵티마이저 상태만 | 낮음 | 낮음 |
| ZeRO-2 | 옵티마이저 + 그래디언트 | 중간 | 중간 |
| ZeRO-3 | 옵티마이저 + 그래디언트 + **파라미터** | 높음 | 높음 |

> 4B student는 ZeRO-2로도 충분하지만, ZeRO-3이 더 많은 배치를 담을 수 있어 throughput에 유리.

### 설치 및 빌드

```bash
# DeepSpeed 설치
pip install deepspeed

# CUDA 연산자 사전 빌드 (H100 권장)
DS_BUILD_OPS=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 \
pip install deepspeed --no-cache-dir

# 설치 확인
ds_report
python -c "import deepspeed; print(deepspeed.__version__)"
```

### ZeRO-3 설정 파일 (`deepspeed_zero3.json`)

```json
{
  "zero_optimization": {
    "stage": 3,

    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,

    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,

    "sub_group_size": 1e9,

    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    }
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,

  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true,
  "communication_data_type": "bf16",

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

#### 주요 파라미터 설명

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `stage` | 3 | 파라미터까지 분산 (최대 절약) |
| `overlap_comm` | true | 통신과 연산 중첩 → 속도 향상 |
| `contiguous_gradients` | true | 연속 메모리 사용 → 통신 효율화 |
| `reduce_bucket_size` | 5e8 | Gradient reduce 버킷 크기 (byte) |
| `stage3_prefetch_bucket_size` | 5e8 | 파라미터 미리 가져오기 크기 |
| `stage3_param_persistence_threshold` | 1e6 | 이 크기 이하 파라미터는 모든 GPU에 유지 (임베딩 등) |
| `stage3_gather_16bit_weights_on_model_save` | true | 저장 시 완전한 가중치 수집 |
| `communication_data_type` | "bf16" | H100에서 BF16 통신으로 대역폭 절약 |

#### CPU Offload 옵션 (GPU 메모리 부족 시)

```json
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
},
"offload_param": {
  "device": "cpu",
  "pin_memory": true
}
```

> 4B student는 32 GPU ZeRO-3로 충분하므로 offload는 기본적으로 `"none"` 사용.  
> Teacher를 같은 프로세스에 로드할 경우 offload 고려 필요.

### ZeRO-3와 Teacher 모델 충돌 방지

ZeRO-3는 학습 대상 모델(student)의 파라미터를 분산한다.  
Teacher 모델을 같은 프로세스에서 `device_map="auto"`로 로드할 경우 DeepSpeed가 teacher 파라미터도 분산하려 시도해 충돌이 발생할 수 있다.

```python
# ❌ 잘못된 방법: teacher도 ZeRO-3에 포함됨
teacher = AutoModelForCausalLM.from_pretrained(...)

# ✅ 올바른 방법 1: exclude_module 사용
import deepspeed
with deepspeed.zero.Init(enabled=False):
    teacher = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-31b-it",
        quantization_config=bnb_config,
        device_map="auto",
    )

# ✅ 올바른 방법 2: 별도 vLLM 서버로 교사 분리 (권장)
# → teacher_model 파라미터 없이 vLLM API 호출로 대체
```

### ZeRO-3 초기화 코드에서 확인

```python
# train.py 상단에서 ZeRO-3 컨텍스트 명시
import deepspeed

# student 모델은 ZeRO-3 분산 대상
student_model = AutoModelForCausalLM.from_pretrained(...)

# teacher 모델은 ZeRO-3 분산 제외
with deepspeed.zero.Init(enabled=False):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        ..., quantization_config=bnb_config, device_map="auto"
    )
```

### ZeRO-3 메모리 절약 효과 (32 GPU 기준)

| 항목 | ZeRO-0 (단일 GPU) | ZeRO-3 (32 GPU) |
|------|-----------------|-----------------|
| 파라미터 (BF16) | 8 GB | 0.25 GB |
| 그래디언트 (BF16) | 8 GB | 0.25 GB |
| 옵티마이저 (FP32) | 32 GB | 1.0 GB |
| **합계** | **48 GB** | **~1.5 GB** |

---

## 7. Accelerate 설정

### accelerate_config.yaml (Node 4-7 학습용)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED

deepspeed_config:
  deepspeed_config_file: deepspeed_zero3.json
  zero3_init_flag: true           # 모델 로드 시점부터 ZeRO-3 적용
  zero3_save_16bit_model: true    # 저장 시 BF16 가중치 수집

num_processes: 32                 # 4 nodes × 8 GPUs
num_machines: 4
machine_rank: 0                   # 각 노드별 0~3 지정
main_process_ip: "NODE4_IP"
main_process_port: 29500
mixed_precision: bf16
gradient_accumulation_steps: 4
```

### 노드별 설정 생성

```bash
# 각 노드(4~7)에서 실행
NODE_RANK=$1  # 0, 1, 2, 3

cat > accelerate_config_node${NODE_RANK}.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: deepspeed_zero3.json
  zero3_init_flag: true
  zero3_save_16bit_model: true
num_processes: 32
num_machines: 4
machine_rank: ${NODE_RANK}
main_process_ip: "NODE4_IP"
main_process_port: 29500
mixed_precision: bf16
EOF
```

---

## 8. 실행 스크립트

### 전체 실행 순서

```bash
# Step 1: Teacher vLLM 서버 기동 (Node 0-3)
# 각 노드에서 실행
bash launch_teacher_vllm.sh <NODE_RANK>  # NODE_RANK=0,1,2,3

# 서버 준비 대기
sleep 60
curl http://node0:8000/health  # 상태 확인

# Step 2: Student 학습 실행 (Node 4-7)
# 각 노드에서 실행
bash launch_student_training.sh <NODE_RANK>  # NODE_RANK=0,1,2,3
```

### launch_student_training.sh

```bash
#!/bin/bash
NODE_RANK=${1:-0}

accelerate launch \
  --config_file accelerate_config_node${NODE_RANK}.yaml \
  --machine_rank ${NODE_RANK} \
  train_kd.py \
    --model_name_or_path google/gemma-4-4b-it \
    --teacher_model_name_or_path google/gemma-4-31b-it \
    --dataset_name your_dataset \
    --output_dir ./output \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --temperature 0.9 \
    --lmbda 0.5 \
    --beta 1.0 \
    --max_new_tokens 512 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --deepspeed deepspeed_zero3.json
```

### torchrun 직접 실행 방식 (대안)

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=${NODE_RANK} \
  --master_addr=NODE4_IP \
  --master_port=29500 \
  --rdzv_backend=c10d \
  train_kd.py \
    --deepspeed deepspeed_zero3.json \
    ...
```

---

## 9. 주요 주의사항

### 모델 관련

- **Flash Attention 2 필수**: `attn_implementation="flash_attention_2"` — H100에서 ~2-3× 처리량 향상
- **TF32 활성화**: H100에서 `tf32=True`로 약 2× 연산 속도 향상 (정확도 거의 동일)
- **Gemma4 vLLM 버전**: vLLM `>= 0.6.0` 필요 (Gemma4 지원 확인)

### ZeRO-3 관련

- **`zero3_init_flag: true`**: 모델 로드 시점부터 파라미터를 분산해 OOM 방지. `false`이면 풀 모델을 먼저 로드 후 분산 → 메모리 순간 피크 발생
- **Teacher 모델 분리**: Teacher는 반드시 `deepspeed.zero.Init(enabled=False)` 컨텍스트 내에서 로드 또는 별도 vLLM 서버로 완전 분리
- **저장 시 가중치 수집**: ZeRO-3에서 저장은 `stage3_gather_16bit_weights_on_model_save: true` 필수. 미설정 시 각 GPU의 파편화된 가중치만 저장됨
- **`reduce_bucket_size` 튜닝**: 너무 작으면 통신 횟수 증가, 너무 크면 메모리 피크 증가. 5e8 (500MB)이 H100 NVLink 환경에서 권장값

### 배치 크기 관련

- **DP 값은 반드시 256의 약수**: 유효한 DP = {1, 2, 4, 8, 16, 32, 64, 128, 256}
- **DP=48 불가**: 256 / 48 = 5.33 (정수 아님)

### 성능 튜닝

| 설정 | 권장값 | 이유 |
|------|-------|------|
| `temperature` | 0.7~1.5 | 너무 낮으면 student 생성 다양성 감소 |
| `lmbda` | 초반 0.7 → 후반 0.3 | 학습 초기 SFT 안정화 후 KD 강화 |
| `learning_rate` | 1e-6~1e-5 | 4B on-policy는 낮은 lr 권장 |
| `gradient_checkpointing` | true | 4B도 활성화 시 메모리 ~30% 절약 |
