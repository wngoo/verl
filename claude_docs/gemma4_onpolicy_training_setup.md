# Gemma4 On-Policy (GRPO) 학습 셋업 가이드

## 1. 핵심 개념: 데이터 포맷은 모델에 무관

parquet의 `prompt` 컬럼은 **모델에 상관없이 동일한 messages 리스트 포맷**이어야 한다.
모델별 chat template 변환(Gemma의 `<start_of_turn>user` 등)은 **tokenizer.apply_chat_template이 자동 처리**한다.

```
# ❌ 잘못된 형태 (plain string)
{"prompt": "What is 2 + 2?"}

# ✅ 올바른 형태 (chat messages list)
{"prompt": [{"role": "user", "content": "What is 2 + 2?"}]}
```

Gemma4용 별도 `convert_prompt`는 필요 없다. `actor_rollout_ref.model.path`만 Gemma4로 지정하면 tokenizer가 자동으로 올바른 포맷으로 변환한다.

---

## 2. 데이터 흐름

```
parquet ["prompt"] = [{"role": "user", "content": "..."}]
          ↓
rl_dataset.py: tokenizer.apply_chat_template(doc["prompt"], ...)
          ↓  (Gemma4 tokenizer가 자동 변환)
<start_of_turn>user\nWhat is 2+2?...<end_of_turn>\n<start_of_turn>model\n
```

오류 원인: `prompt` 컬럼이 plain string이면 `apply_chat_template`이 실패한다. 반드시 messages 리스트여야 한다.

---

## 3. 기존 데이터 변환 (plain string → messages list)

```python
import datasets

ds = datasets.load_dataset("parquet", data_files="train.parquet")["train"]

def convert_prompt(example):
    if isinstance(example["prompt"], str):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
    return example

ds = ds.map(convert_prompt)
ds.to_parquet("train_converted.parquet")
```

---

## 4. 데이터 전처리 스크립트

`examples/data_preprocess/gsm8k.py` 패턴을 따른다. GSM8K는 기존 스크립트 결과물을 그대로 재사용 가능.

```python
# examples/data_preprocess/my_dataset_gemma4.py
import datasets, os

def make_map_fn(split):
    def process_fn(example, idx):
        question = example["question"] + " Let's think step by step."
        solution = example["answer"]
        return {
            "data_source": "my_dataset",
            "prompt": [                           # ← 핵심: messages 리스트
                {"role": "user", "content": question}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {"split": split, "index": idx},
        }
    return process_fn

if __name__ == "__main__":
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train = dataset["train"].map(make_map_fn("train"), with_indices=True)
    test  = dataset["test"].map(make_map_fn("test"),  with_indices=True)
    os.makedirs(os.path.expanduser("~/data/gsm8k"), exist_ok=True)
    train.to_parquet(os.path.expanduser("~/data/gsm8k/train.parquet"))
    test.to_parquet(os.path.expanduser("~/data/gsm8k/test.parquet"))
```

### 필수 parquet 컬럼 구조

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `prompt` | `list[dict]` | `[{"role": "user", "content": "..."}]` |
| `data_source` | `str` | 데이터 출처 식별자 |
| `reward_model` | `dict` | `{"style": "rule", "ground_truth": "..."}` |
| `extra_info` | `dict` | `{"index": idx, ...}` (선택) |

---

## 5. 학습 스크립트 (run_gemma4_gsm8k.sh)

`examples/ppo_trainer/run_qwen2-7b_seq_balance.sh` 기반으로 작성.

```bash
# examples/ppo_trainer/run_gemma4_gsm8k.sh
set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=google/gemma-3-4b-it \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_gemma4_gsm8k' \
    trainer.experiment_name='gemma4-gsm8k-grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@
```

---

## 6. 주의사항

- **vLLM 버전**: Gemma4(`gemma-3-4b-it`)는 vLLM >= 0.8.x 권장
- **`return_raw_chat=True` 필수**: AgentLoop에서 apply_chat_template 처리
- **on-policy이므로 GRPO 추천** (`algorithm.adv_estimator=grpo`): critic 불필요
- **데이터 포맷 변경 불필요**: 기존 GSM8K parquet을 그대로 사용 가능
