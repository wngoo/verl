# On-Policy Distillation: Student Model vLLM Weight Sync 검증

colocate(=verl 용어상 `RolloutMode.HYBRID`) 모드에서 on-policy distillation 학습 시,
FSDP로 학습되는 student 모델의 weight가 vLLM rollout 엔진에 정상적으로 전달되어
초기화되었는지 검증하기 위한 4곳의 breakpoint 위치와 dump 코드.

## 검증 흐름

```
[1] FSDP sender         → [2] vLLM receiver entry
                       ↓
[3a] pre  load_weights  → [3b] post load_weights  → [4] sync done marker
```

각 단계에서 텐서/메타데이터를 `/tmp/verl_dbg/` 에 dump한 뒤 `verify.py`로 일관성을 검사.

---

## 사전 준비

```bash
mkdir -p /tmp/verl_dbg
```

---

## 1) `verl/workers/fsdp_workers.py:876` — FSDP 송신 측

### 대상 라인 (송신 직전)
```python
876:  await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
```

### 검사 포인트
- `per_tensor_param`은 `:832-840`에서 만들어진 **generator** — 그대로 소비하면 dump 후 데이터가 사라짐. list로 materialize한 뒤 동일 list를 update_weights에 넘겨야 함.
- HF 표준 key 인지, GPU 위에 있는지, dtype/shape 정상인지, `base_sync_done`이 첫 호출에 False인지 확인.

### 패치
`:876` 라인을 다음 블록으로 교체:

```python
# === DEBUG DUMP 1: FSDP sender side (replace line 876) ===
import os, torch
_pt_list = list(per_tensor_param)  # generator → list (memory ×2 주의: 첫 sync 1회만)
if torch.distributed.get_rank() == 0 and not self.base_sync_done:
    _summary = []
    _total_numel = 0
    for _name, _tensor in _pt_list:
        _total_numel += _tensor.numel()
        _summary.append({
            "name": _name,
            "shape": tuple(_tensor.shape),
            "dtype": str(_tensor.dtype),
            "device": str(_tensor.device),
            "abs_sum": float(_tensor.detach().float().abs().sum().item()),
            "mean": float(_tensor.detach().float().mean().item()),
        })
    torch.save(
        {
            "stage": "1_fsdp_sender",
            "base_sync_done": self.base_sync_done,
            "peft_config_is_none": peft_config is None,
            "num_params": len(_pt_list),
            "total_numel": _total_numel,
            "summary": _summary,
            # 첫 3개 텐서만 실제 값 저장 (수신 측과 raw 비교용)
            "sample_tensors": {n: t.detach().cpu().clone() for n, t in _pt_list[:3]},
        },
        "/tmp/verl_dbg/1_fsdp_sender.pt",
    )
    print(f"[DBG-1] sender: {len(_pt_list)} params, total_numel={_total_numel}, "
          f"base_sync_done={self.base_sync_done}, first_keys={[s['name'] for s in _summary[:5]]}")
await self.rollout.update_weights(_pt_list, peft_config=peft_config, base_sync_done=self.base_sync_done)
# === END DUMP 1 ===
```

---

## 2) `verl/workers/rollout/vllm_rollout/utils.py:179` — vLLM 수신 측 진입

### 대상 라인
```python
179:  def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
```

### 검사 포인트
- 호출이 도달했는지 (못 도달하면 IPC handshake 단계 문제)
- 인자가 정상인지 (`base_sync_done`, `peft_config`)
- vLLM 모델 클래스, dummy 상태인 모델의 시작 abs_sum (load 후와 비교용)

### 패치
메서드 본문 시작부에 다음을 삽입 (function 시그니처 바로 다음 줄):

```python
def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
    """Update the weights of the rollout model."""
    # === DEBUG DUMP 2: vLLM receiver entry ===
    import os, torch
    _rank = int(os.environ.get("RANK", "0"))
    if _rank == 0:
        _m = self.model_runner.model
        _named = list(_m.named_parameters())
        _pre_sum = sum(p.detach().float().abs().sum().item()
                       for _, p in _named if p.is_floating_point())
        _sample_pre = {n: p.detach().cpu().clone()
                       for n, p in _named[:3] if p.is_floating_point()}
        torch.save(
            {
                "stage": "2_vllm_entry",
                "peft_config_is_none": peft_config is None,
                "base_sync_done": base_sync_done,
                "use_shm": use_shm,
                "model_class": type(_m).__name__,
                "num_named_params": len(_named),
                "pre_load_abs_sum": _pre_sum,
                "sample_pre_load": _sample_pre,
            },
            "/tmp/verl_dbg/2_vllm_entry.pt",
        )
        print(f"[DBG-2] vllm receiver entered: model={type(_m).__name__}, "
              f"base_sync_done={base_sync_done}, pre_load_abs_sum={_pre_sum:.4e}")
    # === END DUMP 2 ===
    from vllm.platforms import current_platform
    ...
```

> **참고:** vLLM engine은 별도 Ray actor 프로세스라 `torch.distributed.get_rank()` 대신
> `os.environ["RANK"]` 또는 `self.local_rank`(이 클래스에 존재) 사용.

---

## 3) `verl/workers/rollout/vllm_rollout/utils.py:264` — `load_weights` 직전·직후 (★ 가장 핵심)

### 대상 라인
```python
263:                logger.info("Loading standard weights (non-FP8, async)")
264:                self.model_runner.model.load_weights(weights)
```

### 검사 포인트
- **직전 (3a):** sender가 보낸 weight가 그대로 도착했는지 (`weights` 리스트 형태,
  key/shape/abs_sum이 1단계 dump와 일치하는지)
- **직후 (3b):** 모델의 실제 파라미터가 진짜로 갱신됐는지 (dummy 때 값 → load 후
  값으로 변했는지)

### 패치
`:263-264`를 다음 블록으로 교체:

```python
                logger.info("Loading standard weights (non-FP8, async)")
                # === DEBUG DUMP 3a: pre load_weights ===
                import os
                _rank = int(os.environ.get("RANK", "0"))
                _weights_list = list(weights) if not isinstance(weights, list) else weights
                if _rank == 0:
                    _summary = [
                        {
                            "name": n,
                            "shape": tuple(t.shape),
                            "dtype": str(t.dtype),
                            "device": str(t.device),
                            "abs_sum": float(t.detach().float().abs().sum().item()),
                        }
                        for n, t in _weights_list
                    ]
                    _m_pre = self.model_runner.model
                    _sample_model_pre = {
                        n: p.detach().cpu().clone()
                        for n, p in list(_m_pre.named_parameters())[:3]
                        if p.is_floating_point()
                    }
                    torch.save(
                        {
                            "stage": "3a_pre_load_weights",
                            "num_weights": len(_weights_list),
                            "weights_summary": _summary,
                            "sample_received": {n: t.detach().cpu().clone()
                                                for n, t in _weights_list[:3]},
                            "sample_model_pre_load": _sample_model_pre,
                        },
                        "/tmp/verl_dbg/3a_pre_load.pt",
                    )
                    print(f"[DBG-3a] pre load_weights: {len(_weights_list)} weights, "
                          f"first_keys={[s['name'] for s in _summary[:5]]}")
                # === END DUMP 3a ===

                self.model_runner.model.load_weights(_weights_list)

                # === DEBUG DUMP 3b: post load_weights ===
                if _rank == 0:
                    _m_post = self.model_runner.model
                    _named_post = list(_m_post.named_parameters())
                    _post_sum = sum(p.detach().float().abs().sum().item()
                                    for _, p in _named_post if p.is_floating_point())
                    _sample_model_post = {
                        n: p.detach().cpu().clone()
                        for n, p in _named_post[:3]
                        if p.is_floating_point()
                    }
                    # pre vs post 같은 layer 비교
                    _diff_report = {}
                    for n, p_post in _sample_model_post.items():
                        if n in _sample_model_pre:
                            _diff_report[n] = float(
                                (p_post.float() - _sample_model_pre[n].float()).abs().mean().item()
                            )
                    torch.save(
                        {
                            "stage": "3b_post_load_weights",
                            "post_load_abs_sum": _post_sum,
                            "sample_model_post_load": _sample_model_post,
                            "pre_post_mean_abs_diff": _diff_report,
                        },
                        "/tmp/verl_dbg/3b_post_load.pt",
                    )
                    print(f"[DBG-3b] post load_weights: post_abs_sum={_post_sum:.4e}, "
                          f"pre_post_diff={_diff_report}")
                # === END DUMP 3b ===
```

> **이 한 곳이 핵심**: `pre_post_mean_abs_diff`가 0이 아니면 vLLM이 실제로 새
> weight로 갱신된 것. 0이면 `load_weights`가 묵묵히 실패한 것.

---

## 4) `verl/workers/fsdp_workers.py:884` — sync 완료 마커

### 대상 라인
```python
884:  self.base_sync_done = True
```

### 검사 포인트
- 첫 sync가 끝까지 도달했는지 (예외 없이) — 1회만 찍히는 마커.
- 이전이 False였고 여기서 True로 바뀌는지 확인.

### 패치
`:884`를 다음 블록으로 교체:

```python
        # === DEBUG DUMP 4: sync done marker ===
        import time, torch
        _was = self.base_sync_done
        self.base_sync_done = True
        if torch.distributed.get_rank() == 0 and _was is False:
            torch.save(
                {
                    "stage": "4_sync_done",
                    "prev_base_sync_done": _was,
                    "now_base_sync_done": self.base_sync_done,
                    "timestamp": time.time(),
                },
                "/tmp/verl_dbg/4_sync_done.pt",
            )
            print(f"[DBG-4] base_sync_done: {_was} -> True (first sync completed)")
        # === END DUMP 4 ===
```

---

## 사후 검증 스크립트

dump 4개를 모두 모은 뒤 일관성 검사. 별도 터미널에서 실행:

```python
# /tmp/verl_dbg/verify.py
import torch

d1 = torch.load("/tmp/verl_dbg/1_fsdp_sender.pt", weights_only=False)
d2 = torch.load("/tmp/verl_dbg/2_vllm_entry.pt", weights_only=False)
d3a = torch.load("/tmp/verl_dbg/3a_pre_load.pt", weights_only=False)
d3b = torch.load("/tmp/verl_dbg/3b_post_load.pt", weights_only=False)
d4 = torch.load("/tmp/verl_dbg/4_sync_done.pt", weights_only=False)

print("=" * 60)
print(f"[1] sender params       : {d1['num_params']}, total_numel={d1['total_numel']:,}")
print(f"[2] vllm entry          : model={d2['model_class']}, base_sync_done={d2['base_sync_done']}")
print(f"    pre-load model sum  : {d2['pre_load_abs_sum']:.4e}  (dummy weights)")
print(f"[3a] received weights   : {d3a['num_weights']}")
print(f"[3b] post-load model sum: {d3b['post_load_abs_sum']:.4e}  (real weights)")
print(f"[3b] pre/post diff      : {d3b['pre_post_mean_abs_diff']}")
print(f"[4] base_sync_done      : {d4['prev_base_sync_done']} -> {d4['now_base_sync_done']}")

# 핵심 체크
assert d1["num_params"] == d3a["num_weights"], "param 개수 불일치 (전송 손실)"
assert d2["pre_load_abs_sum"] != d3b["post_load_abs_sum"], "load 전후 모델이 동일 (load_weights 무시됨)"
assert all(v > 0 for v in d3b["pre_post_mean_abs_diff"].values()), "샘플 layer 변화 없음"

# sender → receiver raw tensor 비교
for (n_s, t_s), (n_r, t_r) in zip(
    d1["sample_tensors"].items(), d3a["sample_received"].items()
):
    assert n_s == n_r, f"key 순서 불일치: {n_s} vs {n_r}"
    diff = (t_s.float() - t_r.float()).abs().mean().item()
    print(f"  [send→recv] {n_s}: mean_abs_diff={diff:.2e}")
    assert diff < 1e-4, f"{n_s} 값이 IPC 중 변형됨"

print("=" * 60)
print("ALL CHECKS PASSED — student model successfully synced to vLLM")
```

---

## 요약 표

| Stage | 파일:라인 | 무엇을 검사 | dump 파일 |
|---|---|---|---|
| 1   | `fsdp_workers.py:876`    | FSDP에서 보내는 텐서 (개수, key, abs_sum, sample) | `1_fsdp_sender.pt` |
| 2   | `utils.py:179`           | vLLM에 호출 도달, dummy 상태 모델 abs_sum         | `2_vllm_entry.pt`  |
| 3a  | `utils.py:264` 직전      | 수신된 `weights`가 sender와 일치                  | `3a_pre_load.pt`   |
| 3b  | `utils.py:264` 직후      | 모델 파라미터가 실제로 변경됨                     | `3b_post_load.pt`  |
| 4   | `fsdp_workers.py:884`    | 전체 sync가 예외 없이 완료                        | `4_sync_done.pt`   |

## 실패 진단 가이드

- **1↔3a 불일치 (param 수 / key 순서 다름)** → IPC 전송 누락 또는 bucket 손실.
  `BucketedWeightSender`/`Receiver`(`bucketed_weight_transfer.py`) 의심.
- **2↔3b의 `pre_load_abs_sum` ≈ `post_load_abs_sum`** → `load_weights`가 묵묵히
  실패. key naming mismatch가 가장 흔한 원인 (FSDP wrapped name vs HF name).
  `convert_weight_keys`(`fsdp_workers.py:792`)와 vLLM 모델의 `load_weights`
  내부 매핑 확인.
- **4가 안 찍힘** → 1~3 사이 어딘가에서 예외. 콘솔 로그/Ray actor 로그 확인.
- **3b의 `pre_post_mean_abs_diff` 모두 0** → 같은 dummy 값을 다시 받았거나
  `load_weights`가 weight를 수용하지 않음. dtype/device mismatch 의심.

## 디버깅 팁

- **rank 0에서만 dump**: 여러 워커가 동시에 dump하면 파일이 덮어써짐. 모든
  스니펫이 rank 0 가드를 둠.
- **vLLM 측은 별도 프로세스**: 일반 `breakpoint()`는 안 잡힘. 대신 `torch.save`
  로 상태를 떠놓고 별도 터미널에서 분석. 또는 `import remote_pdb;
  remote_pdb.set_trace(host='0.0.0.0', port=4444)`.
- **첫 sync에만 찍기**: `if not self.base_sync_done` / `if _was is False` 가드로
  매 step마다 dump되어 디스크가 차지 않게 함.
- **메모리 주의**: `1_fsdp_sender.pt`에서 generator를 list로 변환하므로 첫
  sync 시 일시적으로 텐서 메모리 ×2. 검증 후 제거 권장.
