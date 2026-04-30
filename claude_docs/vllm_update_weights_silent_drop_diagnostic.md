# vLLM `update_weights` Silent Drop 진단 + 누락 보강 패치

FSDP trainer → vLLM rollout 으로 weight를 동기화하는 `update_weights` 경로에서
**일부 weight가 silent하게 누락될 수 있는 가능성**과, 이를 탐지하기 위한 진단 코드,
그리고 누락 가능성이 큰 항목들을 보강하는 패치를 정리한 문서.

---

## 1. 왜 누락이 발생할 수 있나

### 1.1 가장 큰 문제: 누락 검증이 없음

`verl/workers/rollout/vllm_rollout/utils.py` 의 `_update_weights` (수정 전):

```python
else:
    logger.info("Loading standard weights (non-FP8, async)")
    self.model_runner.model.load_weights(weights)
```

vLLM 의 `load_weights()` 는 실제로 로드한 파라미터 이름의 `set[str]`(`loaded_params`)
을 **반환**한다. 그런데 여기서는 **반환값을 버린다.**
vLLM `load_weights` 는 모르는 key 가 들어오면 **에러 없이 그냥 skip** 한다 (예외 없음).

게다가 `_update_weights` 는 **bucket 단위**로 호출되기 때문에, "버킷 전체에 걸쳐 모은
loaded_params ∪ vs. trainer 가 보낸 names" 비교가 어디에도 없다.
→ 일부 weight 이름이 매칭되지 않아도 알 길이 없다.

### 1.2 MoE expert weight 누락 — 알려진 footgun

`utils.py` 원본:
```python
elif use_standard_weight_load:
    # Re-apply here because async IPC weight sync can happen long after init
    # and lose MoE weight_loader attrs.
    patch_vllm_moe_model_weight_loader(self.model_runner.model)
```

`mlp.experts.w13_weight`, `w2_weight` 는 **커스텀 `weight_loader` attr** 이 있어야만
로드된다 (`verl/utils/vllm/patch.py:140-142`). 이게 없으면 vLLM 의 default loader 가
이름을 못 알아보고 skip 한다.

문제는 이 patch 가 `use_standard_weight_load` 분기에서만 다시 적용된다는 점이다.
**LoRA + `base_sync_done=True`** 또는 **FP8/QAT/ModelOpt** 경로에서는 재적용되지
않으므로, 그런 경로에서 MoE expert 가 silent 하게 안 업데이트될 가능성이 있다.

### 1.3 `convert_weight_keys` 의 regex mismatch

`verl/utils/model.py:242-260` — HF 의 `_checkpoint_conversion_mapping` 기반 regex 치환.
pattern 이 안 맞으면 원본 key 그대로 통과하고, vLLM 쪽은 그 key 를 모르므로 silent skip.

### 1.4 FSDP `state_dict()` 가 빠뜨리는 것들

`fsdp_workers.py` 의 `params = self.actor_module_fsdp.state_dict()` 는:

- **non-persistent buffer 는 제외** (`register_buffer(..., persistent=False)`)
- **tied weight** 는 한쪽만 포함될 수 있음 (예: `lm_head.weight` 와 `embed_tokens.weight`)

### 1.5 dtype/shape mismatch 시의 silent skip

`vllm_rollout.py` 위 주석:
> we should not force cast weight here because some parameters
> (such as moe gate) have to keep fp32 precision

dtype 이 vLLM 쪽 expected 와 안 맞으면 vLLM 쪽 `weight_loader` 의 `assert` 또는
dtype check 에서 실패할 수 있고, 일부 loader 는 그냥 skip 한다.

---

## 2. 적용한 패치

### 2.1 `verl/workers/rollout/vllm_rollout/utils.py`

#### `_update_weights` — `loaded set` 반환

```python
def _update_weights(
    self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool
) -> Optional[set]:
    if peft_config and base_sync_done:
        ...
        self.add_lora(lora_request)
        # LoRA path: add_lora doesn't return a loaded set; coverage check N/A.
        return None
    else:
        if is_fp8_model(self.model_runner.vllm_config):
            ...
            loaded_params = load_quanted_weights(weights, self.model_runner)
            return set(loaded_params) if loaded_params is not None else set()
        else:
            loaded = self.model_runner.model.load_weights(weights)
            # vLLM's load_weights returns the set of param names actually loaded.
            # Older versions may return None — treat as empty so the diagnostic
            # in update_weights_from_ipc fails closed (warns about everything).
            return set(loaded) if loaded is not None else set()
```

`vLLMOmniColocateWorkerExtension._update_weights` 에도 동일하게 적용.

#### `update_weights_from_ipc` — 항상 MoE patch + 버킷 누적 진단

```python
# Always re-apply MoE weight_loader patch. Async IPC weight sync can happen long
# after init and lose MoE weight_loader attrs, causing expert weights (w13/w2) to
# be silently skipped. Safe no-op for non-MoE models. Previously gated on the
# standard-load branch only — but QAT/ModelOpt MoE models need it too.
patch_vllm_moe_model_weight_loader(self.model_runner.model)

if self._is_qat_model:
    ...
elif self._is_modelopt_qat:
    ...

assert self.device is not None
receiver = BucketedWeightReceiver(...)

# Diagnostic: vLLM's load_weights silently skips unknown keys. Without aggregation
# across buckets we cannot tell if any sent weight was dropped. Track:
#   all_received - all_loaded  -> sent but vLLM didn't acknowledge -> SILENT DROP.
# Disable with VERL_VLLM_UPDATE_WEIGHTS_DIAGNOSTIC=0.
diagnostic_enabled = os.environ.get("VERL_VLLM_UPDATE_WEIGHTS_DIAGNOSTIC", "1") == "1"
all_received_keys: set = set()
all_loaded_keys: set = set()

def _on_bucket(weights):
    if diagnostic_enabled:
        all_received_keys.update(name for name, _ in weights)
    loaded = self._update_weights(
        weights, peft_config=peft_config, base_sync_done=base_sync_done
    )
    if diagnostic_enabled and loaded is not None:
        all_loaded_keys.update(loaded)

receiver.receive_weights(on_bucket_received=_on_bucket)

# LoRA's add_lora has different semantics (no loaded-set) — skip the diff there.
if diagnostic_enabled and not (peft_config and base_sync_done):
    unmatched = all_received_keys - all_loaded_keys
    if unmatched:
        sample = sorted(unmatched)[:10]
        logger.warning(
            "update_weights: %d of %d sent keys were NOT acknowledged by vLLM "
            "load_weights — those weights were silently dropped. Sample: %s",
            len(unmatched),
            len(all_received_keys),
            sample,
        )
    else:
        logger.info(
            "update_weights: all %d sent keys acknowledged by vLLM.",
            len(all_received_keys),
        )
```

요점:
- `patch_vllm_moe_model_weight_loader` 를 `if/elif` 체인 **밖**으로 빼서 모든 경로에
  적용. 함수 자체가 MoE 모델 아니면 early-return 하므로 안전.
- `_on_bucket` closure 안에서 `all_received_keys`, `all_loaded_keys` 두 set 을 누적.
- 마지막 bucket 처리 후 `all_received - all_loaded` = vLLM 이 ack 하지 않은 키 =
  **silent drop**. 발견되면 WARN + sample 10개 출력.

`vLLMOmniColocateWorkerExtension.update_weights_from_ipc` 에도 동일 패턴 적용.

### 2.2 `verl/workers/fsdp_workers.py`

```python
else:
    params = self.actor_module_fsdp.state_dict()
    # state_dict() omits non-persistent buffers (registered with persistent=False).
    # If the trainer model has any (e.g. RoPE inv_freq, RMSNorm caches in some
    # implementations), they would never be transferred. Add them explicitly so
    # vLLM sees the full set; named_buffers() returns persistent + non-persistent.
    for buf_name, buf in self.actor_module_fsdp.named_buffers():
        if buf_name not in params:
            params[buf_name] = buf
```

PyTorch 의 `state_dict()` 는 `register_buffer(..., persistent=False)` 를 제외한다.
`named_buffers()` 는 persistent / non-persistent 모두 반환하므로, `state_dict` 에
없는 buffer 만 보충하는 형태로 안전하게 합칠 수 있다.

---

## 3. 사용법 / 확인 방법

학습 돌릴 때 로그 확인:

- **정상**: `update_weights: all 387 sent keys acknowledged by vLLM.`
- **누락**:
  ```
  update_weights: 12 of 387 sent keys were NOT acknowledged by vLLM load_weights
    — those weights were silently dropped.
    Sample: ['model.layers.0.mlp.experts.w13_weight', ...]
  ```

샘플로 잡히는 prefix 로 원인 추정:

| Sample 패턴 | 의심되는 원인 |
|---|---|
| `mlp.experts.w13_weight`, `w2_weight` | MoE patch 누락 (이번 패치로 해결) |
| `lm_head.*`, `embed_tokens.*` | tied weight 문제 |
| `*inv_freq`, RoPE buffer | non-persistent buffer (이번 패치로 해결) |
| 그 외 | `convert_weight_keys` regex mismatch |

진단 끄기:
```bash
export VERL_VLLM_UPDATE_WEIGHTS_DIAGNOSTIC=0
```

---

## 4. 추가로 검토해볼 것

- **Sender-side 진단**: `fsdp_workers.py` 에서 `params` 의 key 개수 / 이름을 로깅하고
  receiver 쪽 ack 와 양방향 검증.
- **양쪽 logit 비교**: trainer 모델과 vLLM 모델에 같은 입력을 넣어 logit 차이를 매
  step / 가끔 확인. update 직후 차이가 크면 누락 의심.
- **MoE 모델일 때**: `inner_model.layers[0].mlp.experts.w13_weight` 를 update 전후로
  hash 찍어 변하는지 확인.
- **megatron path**: `verl/workers/megatron_workers.py` 의 `update_weights` 도 동일한
  패턴 검토 필요.

---

## 5. 변경 파일 목록

- `verl/workers/rollout/vllm_rollout/utils.py`
  - `vLLMColocateWorkerExtension._update_weights` — return loaded set
  - `vLLMColocateWorkerExtension.update_weights_from_ipc` — always-on MoE patch + diagnostic
  - `vLLMOmniColocateWorkerExtension._update_weights` — return loaded set
  - `vLLMOmniColocateWorkerExtension.update_weights_from_ipc` — diagnostic
- `verl/workers/fsdp_workers.py`
  - `AsyncActorRolloutRefWorker.rollout_mode` (의 base) — non-persistent buffer 보충
