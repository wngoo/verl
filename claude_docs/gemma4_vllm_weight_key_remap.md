# Gemma 4 vLLM weight key remap — silent drop 수정

FSDP trainer → vLLM rollout 으로 weight를 동기화할 때 Gemma 4 멀티모달 모델은
**HF 학습 측 key 이름**과 **vLLM 측 expected key 이름**의 layout이 달라서
`load_weights()` 가 모든 weight를 silent하게 drop한다. 본 문서는 그 원인과
`verl/workers/rollout/vllm_rollout/utils.py` 에 적용한 수정 내용을 정리한다.

관련: `vllm_update_weights_silent_drop_diagnostic.md` (silent drop 탐지 진단의
상위 프레임워크). 본 문서는 그 진단으로 드러난 Gemma 4 케이스의 구체적 원인과
패치를 다룬다.

---

## 1. 증상

`VERL_VLLM_KEY_DIAGNOSTIC=1` 상태에서 Gemma 4 학습을 돌리면 아래 로그가 찍힌다.

```
[KEY_DIAG] vLLM model class: Gemma4ForConditionalGeneration
[KEY_DIAG] vLLM expects 943 params
[KEY_DIAG] vLLM expected prefixes (top-2):
  ['audio_tower.layers', ..., 'language_model.model', 'vision_tower.encoder', ...]
[KEY_DIAG] received first bucket: 537 keys
[KEY_DIAG] received prefixes (top-2): ['model.language_model']
[KEY_DIAG] received sample: ['model.language_model.embed_tokens.weight', ...]
[KEY_DIAG] exact-match in this bucket: 0 of 537 received keys exist verbatim in vLLM
update_weights: 1952 of 1952 sent keys were NOT acknowledged by vLLM load_weights
  — those weights were silently dropped
```

**전송한 1952개 key 모두가 vLLM에 의해 drop 됨.** 즉 vLLM은 학습된 weight를
하나도 받지 못하고 초기 weight 그대로 inference를 수행하게 된다. 텍스트만
사용하는 경우라도 language_model 측 key가 0/537 매칭이라 결과 품질이 무너진다.

---

## 2. 원인: Gemma 4 멀티모달 트리 layout 불일치

| HF (학습 측 보내는 key)            | vLLM (`Gemma4ForConditionalGeneration` 기대 key) |
|------------------------------------|---------------------------------------------------|
| `model.language_model.embed_tokens.weight` | `language_model.model.embed_tokens.weight` |
| `model.language_model.layers.0.*`  | `language_model.model.layers.0.*`                 |
| `model.audio_tower.*`              | `audio_tower.*`                                   |
| `model.vision_tower.*`             | `vision_tower.*`                                  |
| `model.embed_audio.*`              | `embed_audio.*`                                   |
| `model.embed_vision.*`             | `embed_vision.*`                                  |
| `lm_head.weight`                   | (없음 — Gemma는 tied embeddings)                  |
| `*.input_max`/`*.input_min`/<br>`*.output_max`/`*.output_min` | (없음 — quantization observer stats)|

핵심 차이는 두 가지.

1. **`language_model` prefix swap**: HF는 최상위 `model` 아래에 `language_model`이
   있고, vLLM은 최상위 `language_model` 아래에 다시 `model`이 있다. 즉
   `model.language_model.X` ↔ `language_model.model.X` 의 swap.
2. **multimodal tower / embed의 `model.` prefix 제거**: `audio_tower`,
   `vision_tower`, `embed_audio`, `embed_vision` 은 vLLM 측에서 최상위에 있다.
   HF의 `model.` prefix를 떼야 한다.

추가로 vLLM이 받지 않는 항목은 drop 해야 dropped-key warning을 깨끗이 비울 수
있다.

- `lm_head.weight` — Gemma는 tied embedding이므로 vLLM이 별도 param으로 등록하지
  않는다 (`vLLM has ... lm_head=False` 로 KEY_DIAG가 알려준다).
- `*_max` / `*_min` — activation quantization observer stat. 일반 weight가 아님.

---

## 3. 수정 (`verl/workers/rollout/vllm_rollout/utils.py`)

### 3.1 module-level helper 추가

`monkey_patch_compute_logits` 직후, 첫 번째 클래스 정의 바로 위에 추가.

```python
def _is_gemma4_vllm_model(model_obj) -> bool:
    """True if the vLLM model class is Gemma 4 (multimodal)."""
    if model_obj is None:
        return False
    return type(model_obj).__name__.startswith("Gemma4")


def _remap_gemma4_keys(
    weights: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Rename HF-side Gemma 4 keys to vLLM's Gemma4ForConditionalGeneration layout."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name.endswith(("_max", "_min")):
            continue
        if name == "lm_head.weight":
            continue

        new_name = name
        if name.startswith("model.language_model."):
            new_name = "language_model.model." + name[len("model.language_model.") :]
        elif name.startswith(
            ("model.audio_tower.", "model.vision_tower.",
             "model.embed_audio.", "model.embed_vision.")
        ):
            new_name = name[len("model.") :]
        out.append((new_name, tensor))
    return out
```

### 3.2 colocate path (`vLLMColocateWorkerExtension._on_bucket`)

`_on_bucket` 함수 진입 직후, diagnostic + load 보다 앞에서 remap 호출.

```python
def _on_bucket(weights):
    if not hasattr(self, "_gemma4_remap_needed"):
        self._gemma4_remap_needed = _is_gemma4_vllm_model(self.model_runner.model)
    if self._gemma4_remap_needed:
        weights = _remap_gemma4_keys(weights)
    # ...기존 diagnostic + _update_weights 호출...
```

`_gemma4_remap_needed` 는 instance에 caching하여 bucket마다 model class 검사를
반복하지 않는다.

### 3.3 omni path (`vLLMOmniColocateWorkerExtension._on_bucket`)

omni는 model 객체 fetch 경로가 약간 달라(`model_runner.model` 또는 `self.model`)
그것에 맞춰 처리.

```python
def _on_bucket(weights):
    if not hasattr(self, "_gemma4_remap_needed"):
        try:
            model_obj_local = (
                getattr(self, "model_runner", None) and self.model_runner.model
            ) or getattr(self, "model", None)
        except Exception:
            model_obj_local = None
        self._gemma4_remap_needed = _is_gemma4_vllm_model(model_obj_local)
    if self._gemma4_remap_needed:
        weights = _remap_gemma4_keys(weights)
    # ...기존 diagnostic + _update_weights 호출...
```

### 3.4 다른 모델은 영향 없음

Gemma 4가 아닌 모든 모델에서는 `_gemma4_remap_needed=False` 가 cache되어 분기
자체를 통과만 한다. 기존 동작은 변하지 않는다.

---

## 4. 검증 방법

### 4.1 KEY_DIAG 로그 확인

수정 후 학습을 다시 돌리면 첫 번째 bucket에서 아래처럼 바뀌어야 한다.

```
[KEY_DIAG] received prefixes (top-2): ['language_model.model']
[KEY_DIAG] received sample: ['language_model.model.embed_tokens.weight', ...]
[KEY_DIAG] exact-match in this bucket: 537 of 537   # ← 0 → 537 으로 회복
```

### 4.2 silent drop warning 확인

`VERL_VLLM_UPDATE_WEIGHTS_DIAGNOSTIC=1` 일 때 다음 로그가 떠야 한다.

```
update_weights: all N sent keys acknowledged by vLLM.
```

이전의 `1952 of 1952 sent keys were NOT acknowledged` 경고가 사라져야 한다.

### 4.3 만약 일부가 여전히 drop된다면

남은 dropped key sample을 KEY_DIAG가 출력해 준다. 이 문서의 매핑 표에 없는
prefix가 있으면 `_remap_gemma4_keys` 의 분기에 추가해야 한다.

---

## 5. 텍스트만 쓰는 경우 (text-only inference)

audio_tower / vision_tower / embed_audio / embed_vision weight는 텍스트
inference 시 사용되지 않으므로 functional하게는 skip해도 무방하다. 그러나
**language_model의 prefix swap은 필수**이다 (텍스트 생성에 직접 사용됨).

본 패치는 두 종류 모두 정상적으로 remap한다 — 텍스트 전용이라도 audio/vision
weight를 remap해서 보내는 비용은 작고, dropped-key warning을 깨끗이 비우는
이점이 있다.

---

## 6. 참고

- 진단 시스템 자체에 대한 설명: `vllm_update_weights_silent_drop_diagnostic.md`
- Gemma 4 학습 셋업: `gemma4_onpolicy_training_setup.md`
- weight transfer flow: `vllm_colocate_launch_flow.md`
