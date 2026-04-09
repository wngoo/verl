---
date: 2026-04-09
topic: on-policy KD 학습 시 SingleTurnAgentLoop가 실행되는 이유와 실행 경로
tags: [agent_loop, on_policy_kd, distillation, rollout, ray_trainer]
---

# On-Policy KD에서 SingleTurnAgentLoop가 실행되는 이유

## 개요

`run_qwen_gsm8k.sh`를 이용한 on-policy knowledge distillation 학습 시,
`verl/experimental/agent_loop/single_turn_agent_loop.py`의 `run()` 함수가 실행된다.
이 문서는 왜 실행되는지와 전체 실행 경로를 설명한다.

## 실행 경로

```
run_qwen_gsm8k.sh
  → python -m verl.trainer.main_ppo
    → ray_trainer.py: AgentLoopManager.create(...)
      → AgentLoopWorker가 각 배치의 rollout 처리
        → 각 샘플에 agent_name = "single_turn_agent" (default) 할당
          → SingleTurnAgentLoop.run() 호출 (student 모델 응답 생성)
```

## 핵심 내용

### 왜 SingleTurnAgentLoop인가?

기본값 설정:
- `verl/trainer/config/rollout/rollout.yaml:229` → `default_agent_loop: single_turn_agent`
- `verl/workers/config/rollout.py:100` → `default_agent_loop: str = "single_turn_agent"`
- `run_qwen_gsm8k.sh`에서 이 값을 덮어쓰지 않음

`SingleTurnAgentLoop`는 `@register("single_turn_agent")` 데코레이터로 등록
(`single_turn_agent_loop.py:33`).

배치에 `agent_name`이 없으면 `agent_loop.py:537`에서 default를 자동으로 할당:
```python
if "agent_name" not in batch.non_tensor_batch:
    default_agent_loop = config.agent.default_agent_loop
    batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)
```

### SingleTurnAgentLoop의 역할

- tool 호출 없이 **단일 턴 생성(single-turn generation)** 수행
- vLLM 또는 SGLang을 통해 **student 모델의 응답을 생성**
- 생성된 응답은 teacher 모델로 전달 → distillation loss (KL divergence 등) 계산
- GSM8K처럼 단순 수학 문제는 multi-turn이나 tool use가 불필요하므로 적합

### AgentLoop 종류와 선택 기준

| AgentLoop 이름 | 등록 키 | 용도 |
|---|---|---|
| `SingleTurnAgentLoop` | `single_turn_agent` | 단일 턴 응답 생성 (기본값) |
| `ToolAgentLoop` | `tool_agent` | 툴 호출이 필요한 agentic 태스크 |
| `DiffusionSingleTurnAgentLoop` | `diffusion_single_turn_agent` | Diffusion 모델용 단일 턴 |

커스텀 AgentLoopManager도 설정 가능:
```yaml
actor_rollout_ref.rollout.agent.agent_loop_manager_class: my.module.CustomManager
```

## 상세 설명

### ray_trainer.py에서 AgentLoopManager 초기화

`ray_trainer.py:857-880`:
```python
# Support custom AgentLoopManager via config
manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
if manager_class_fqn:
    AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
else:
    from verl.experimental.agent_loop import AgentLoopManager

self.async_rollout_manager = AgentLoopManager.create(
    config=self.config,
    worker_group=self.actor_rollout_wg,
    rollout_resource_pool=actor_rollout_resource_pool,
    reward_loop_worker_handles=reward_loop_worker_handles,
    teacher_model_manager=self.teacher_model_manager,  # on-policy KD 시 teacher 전달
)
```

### On-Policy KD 특이사항

`run_qwen_gsm8k.sh`에서 `distillation.enabled=True`로 설정하면:
- `ray_trainer.py:844-854`에서 `TeacherModelManager`도 함께 초기화됨
- `async_rollout_manager`에 `teacher_model_manager`가 주입됨
- rollout 이후 teacher가 동일 프롬프트+응답에 대한 log_prob 계산 → distillation loss 산출

## 관련 파일/코드

| 파일 | 역할 |
|---|---|
| `examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh` | 학습 실행 스크립트 |
| `verl/trainer/ppo/ray_trainer.py:857-880` | AgentLoopManager 초기화 |
| `verl/experimental/agent_loop/agent_loop.py:536-538` | default agent_name 할당 |
| `verl/experimental/agent_loop/single_turn_agent_loop.py:33` | `@register("single_turn_agent")` |
| `verl/trainer/config/rollout/rollout.yaml:229` | default_agent_loop 기본값 설정 |
| `verl/workers/config/rollout.py:100` | RolloutConfig의 default_agent_loop 필드 |
| `verl/experimental/agent_loop/__init__.py` | AgentLoop 클래스 export |

## 참고 사항

- `trainer.use_legacy_worker_impl=disable`로 설정 시 새로운 engine worker 구현 사용
  (기존 `fsdp_workers.py` 대신 `engine_workers.py`)
- Agent loop 변경이 필요한 경우(예: tool use 필요) `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent` 설정
- 관련 문서: `claude_docs/on_policy_knowledge_distillation.md`
