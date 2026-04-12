# 멀티턴 데이터에서 On-Policy KD 마스킹 및 Loss 구조

> 분석 기준 스크립트: `examples/on_policy_distillation_trainer/run_gemma4_e4b_opd_gemma4_e4b_on.sh`
> 설정: `loss_mode=k1`, `use_policy_gradient=True`, `use_task_rewards=False`

---

## 목차

1. [데이터 포맷과 Prompt/Response 분리](#1-데이터-포맷과-promptresponse-분리)
2. [토큰화 및 마스킹](#2-토큰화-및-마스킹)
3. [Loss 계산 상세](#3-loss-계산-상세)
4. [전체 학습 흐름](#4-전체-학습-흐름)
5. [결론 요약](#5-결론-요약)
6. [코드 위치 참조](#6-코드-위치-참조)

---

## 1. 데이터 포맷과 Prompt/Response 분리

### 입력 데이터 포맷 예시

```json
{
  "messages": [
    {"role": "system",    "content": "You are a function calling AI model..."},
    {"role": "user",      "content": "Create a todo list..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>todo(...)</tool_call>"},
    {"role": "tool",      "content": "{\"todos\": [...]}"}
  ]
}
```

스크립트에서 `data.prompt_key="prompt"` 이고 데이터의 key가 `"messages"`인 경우:

```bash
data.prompt_key="prompt"   # 데이터셋의 "prompt" 필드 → messages 리스트 전체
```

### apply_chat_template 처리

`RLHFDataset.__getitem__`에서 messages 전체를 `raw_prompt`로 반환하고, AgentLoop에서 아래와 같이 처리:

```python
tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
```

`add_generation_prompt=True`이면 마지막 메시지 이후에 모델 생성 프롬프트(`<start_of_turn>model\n` 등)가 자동 추가된다.

### Prompt / Response 경계

```
┌──────────────────────────────────────────────────────────────┐
│                       PROMPT (컨텍스트)                       │
│  [system]    You are a function calling AI model...          │
│  [user]      Create a todo list...                           │
│  [assistant] <think>...</think> <tool_call>...</tool_call>   │  ← 기존 중간 assistant 턴
│  [tool]      {"todos": [...]}                                │  ← 툴 응답
│  <start_of_turn>model\n                                      │  ← generation prompt
├──────────────────────────────────────────────────────────────┤
│                   RESPONSE (on-policy 생성)                   │
│  새로 생성되는 마지막 assistant 턴                              │
│  ex) "The todo list has been created. Here are priorities..."│
└──────────────────────────────────────────────────────────────┘
```

**messages의 마지막이 `tool` 턴으로 끝나므로, student가 on-policy로 생성하는 것은 그 다음 assistant 턴 하나뿐이다.**

---

## 2. 토큰화 및 마스킹

### response_mask 계산

```python
# verl/trainer/ppo/ray_trainer.py:119-134
def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)           # 새로 생성된 토큰 수
    attention_mask = data.batch["attention_mask"] # [prompt + response] 전체
    return attention_mask[:, -response_length:]   # 뒤쪽 response 부분만 추출
```

### 마스킹 결과

```
전체 시퀀스 (attention_mask):
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 0 0]
 |←──────── PROMPT ──────────→| |←─ RESPONSE →|
  system+user+assistant+tool       새로 생성된 턴

response_mask = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 0 0]
                                               ↑ 이 부분만 1
```

| 토큰 종류 | response_mask | Loss |
|---|---|---|
| system 토큰 | 0 | ❌ 없음 |
| user 토큰 | 0 | ❌ 없음 |
| **기존 중간 assistant 턴** | **0** | **❌ 없음** |
| tool_response 토큰 | 0 | ❌ 없음 |
| **새로 생성된 assistant 턴** | **1** | **✅ KD loss 적용** |
| padding 토큰 | 0 | ❌ 없음 |

> **핵심**: 멀티턴 대화의 기존 중간 assistant 응답들은 loss에서 완전히 제외된다.  
> 새로 생성된 마지막 assistant 턴에만 KD loss가 적용된다.

---

## 3. Loss 계산 상세

### Step 1 — k1 loss 계산 (per response token)

```python
# verl/trainer/distillation/losses.py:341-353
student_log_probs = model_output["log_probs"]          # log π_student(token)
teacher_log_probs = data["teacher_logprobs"].squeeze(-1) # log π_teacher(token)

distillation_losses = kl_penalty(
    logprob=student_log_probs,
    ref_logprob=teacher_log_probs,
    kl_penalty="k1"   # k1 = log π_student - log π_teacher
)
# shape: (bsz, resp_len)
```

k1은 단일 샘플 KL divergence 추정치:

```
k1_token = log π_student(token) - log π_teacher(token)

  k1 > 0: student가 teacher보다 해당 토큰을 과신 → 억제 필요
  k1 < 0: teacher가 student보다 해당 토큰을 더 선호 → 장려 필요
  k1 = 0: student와 teacher가 동일한 확률 할당
```

### Step 2 — Policy gradient advantage로 변환

```python
# verl/trainer/distillation/losses.py:253-269
distillation_loss, pg_metrics = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=-distillation_losses.detach(),  # ← 부호 반전: k1을 negative reward로
    response_mask=response_mask,
    ...
)
```

```
advantage = -k1 = log π_teacher - log π_student

  k1 > 0 (student 과신):  advantage < 0 → 해당 토큰 생성 확률 감소
  k1 < 0 (teacher 선호):  advantage > 0 → 해당 토큰 생성 확률 증가
  → student 분포가 teacher 분포에 가까워지는 방향으로 학습
```

이 방식은 [On-Policy Distillation (Thinking Machines AI)](https://thinkingmachines.ai/blog/on-policy-distillation/) 에서 제안된 방법이다.

### Step 3 — task reward 없이 KD loss만 사용

```python
# verl/trainer/distillation/losses.py:207-215
if not distillation_loss_config.use_task_rewards:  # use_task_rewards=False
    policy_loss = 0.0                               # PPO policy loss 제거

policy_loss += distill_loss * 1.0                   # KD loss만 남음
```

스크립트에서 `distillation.distillation_loss.use_task_rewards=False`이므로 환경 reward 없이 순수 KD loss로만 학습한다.

---

## 4. 전체 학습 흐름

```
[1] 데이터 로드
    messages (system+user+assistant+tool) → prompt 토큰화
    마지막이 tool 턴 → student가 다음 assistant 턴 생성

[2] Student Rollout (on-policy 생성)
    student vLLM이 마지막 assistant 턴을 새로 생성
    생성된 토큰 = response (KD loss 대상)

[3] Teacher Inference
    teacher vLLM이 response 토큰 위치에서
    log π_teacher(token | prompt + 앞 response 토큰들) 계산
    ※ teacher는 새로 생성된 response만 처리 (prompt는 공유 컨텍스트)

[4] KD Loss 계산 (response_mask=1인 토큰에만)
    k1 per token = log π_student - log π_teacher
    advantage = -k1 (teacher 선호 방향)
    GRPO style policy gradient update

[5] Actor Update (FSDP)
    student가 teacher 분포에 가까워지도록 파라미터 업데이트
    prompt 부분(기존 중간 턴 포함)은 loss 기여 없음
```

---

## 5. 결론 요약

| 항목 | 동작 |
|---|---|
| KD가 적용되는 턴 | **마지막 assistant 턴 하나만** (새로 on-policy 생성된 것) |
| 기존 중간 assistant 턴 | **프롬프트 컨텍스트로만 활용**, loss 없음 |
| `<think>` 블록 (새로 생성된 것) | **KD loss 포함** (response 토큰이므로) |
| tool_call JSON (새로 생성된 것) | **KD loss 포함** (response 토큰이므로) |
| tool_response 내용 | **loss 없음** (프롬프트 부분) |
| loss 종류 | k1 추정 KL → GRPO advantage로 변환 |
| task reward | 없음 (`use_task_rewards=False`) |

### 멀티턴 활용 가이드

- messages의 **마지막을 항상 user 또는 tool 턴으로 끝내야** 한다
  - 마지막이 assistant 턴이면, 그 턴도 프롬프트에 포함되어 generation 대상이 달라짐
- 대화 히스토리가 길수록 `MAX_PROMPT` 한계에 주의
- 중간 assistant 턴의 tool_call 패턴은 학습에 직접 영향을 주지 않지만, student의 tool_call 생성 능력에 영향을 주는 인-컨텍스트 예시가 됨

---

## 6. 코드 위치 참조

| 역할 | 파일 | 위치 |
|---|---|---|
| 데이터셋 로드 및 토큰화 | `verl/utils/dataset/rl_dataset.py` | `RLHFDataset.__getitem__:359` |
| response_mask 계산 | `verl/trainer/ppo/ray_trainer.py` | `compute_response_mask:119` |
| k1 loss 계산 | `verl/trainer/distillation/losses.py` | `compute_distillation_loss_reverse_kl_estimator:322` |
| policy gradient 변환 | `verl/trainer/distillation/losses.py` | `distillation_loss:253` |
| task reward 분기 | `verl/trainer/distillation/losses.py` | `distillation_ppo_loss:207` |
