# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Agent instructions for verl also apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Installing the package

```bash
# Development install with test + rollout backend
pip install -e .[test,vllm]    # vLLM backend
pip install -e .[test,sglang]  # SGLang backend
pip install -e .[test,mcore]   # Megatron-Core backend

# Optional GPU optimizations
pip install -r requirements-cuda.txt
```

### Linting and formatting

Ruff (line length 120) is the primary linter/formatter; mypy handles type checking. Both run via pre-commit:

```bash
pre-commit run --all-files
pre-commit run ruff --all-files --show-diff-on-failure --color=always
```

Custom pre-commit hooks also enforce docstring coverage, license headers, device API usage, and naming conventions (`verl` not `veRL`; `SGLang`/`sglang` not variants). The hook `autogen-trainer-cfg` auto-generates `verl/trainer/config/_generated_*.yaml` — do not edit these files manually.

### Running tests

```bash
# CPU-only unit tests (fast; default in CI on every PR)
pytest -s -x --asyncio-mode=auto tests/ -k "*_on_cpu"

# Run a single test file
pytest -s -x --asyncio-mode=auto tests/test_protocol_on_cpu.py

# GPU tests (require hardware)
pytest -s -x --asyncio-mode=auto tests/workers/
```

Tests are organized to mirror the package structure: `tests/trainer/` covers `verl/trainer/`, `tests/workers/` covers `verl/workers/`, etc. E2E and distributed tests live under `tests/special_e2e/` and `tests/special_distributed/`.

### Commit messages

Add attribution using commit trailers:

```text
Your commit message here

Co-authored-by: Claude
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## 3. Architecture Overview

Verl (Volcano Engine Reinforcement Learning) is a post-training RL framework for LLMs implementing the **HybridFlow** programming model (EuroSys 2025). It separates *control flow* (RL algorithm logic) from *computation flow* (distributed neural network execution).

### Core concepts

- **Controller**: A single Python process that runs the RL algorithm loop (e.g., PPO: rollout → advantage computation → policy update). Lives in `verl/single_controller/`.
- **Worker groups**: Ray actor pools that perform distributed computation. The controller issues remote calls to workers; workers never call back into the controller.
- **DataProto** (`verl/protocol.py`): The shared data structure passed between controller and workers. Built on TensorDict.

### Main worker types (`verl/workers/`)

| Worker | Role |
|---|---|
| `ActorRolloutRefWorker` | Generates sequences (rollout) and computes log-probs for the policy and reference model |
| `CriticWorker` | Computes value estimates |
| `RewardManager` (`reward_manager/`) | Rule-based or model-based reward computation |

Each worker has FSDP (`fsdp_workers.py`) and Megatron-LM (`megatron_workers.py`) backend variants.

### Rollout engines (`verl/workers/rollout/`)

Pluggable generation backends: vLLM (`vllm_rollout/`), SGLang (`sglang_rollout/`), HuggingFace Transformers (`hf_rollout.py`), and a naive reference implementation.

### Training entry points

- `verl/trainer/main_ppo.py` — main PPO/GRPO/DAPO/VAPO entry point (Hydra-configured)
- `verl/trainer/sft_trainer.py` — supervised fine-tuning

```bash
# Example launch
python -m verl.trainer.main_ppo \
  --config-path examples/ppo_trainer \
  --config-name run_qwen2-7b \
  trainer.output_dir=/tmp/verl-output
```

### Configuration system

Hydra YAML configs live in `verl/trainer/config/`. Algorithm-specific configs (PPO, GRPO, DPPO, etc.) are composable overlays on a base config. Auto-generated configs (`_generated_*.yaml`) are produced by a pre-commit hook from the dataclass definitions in `verl/trainer/config/`.

### Key supporting modules

- `verl/models/` — HF Transformer and Megatron-Core model loading/extensions, diffusion models
- `verl/checkpoint_engine/` — model save/load
- `verl/interactions/` — multi-turn conversation and tool-use support
- `verl/tools/` — math verification, web search, and VLM tools used as reward signals
- `verl/utils/` — device helpers, functional ops, logging
- `verl/experimental/` — async RL, off-policy learning, agentic RL (not production-ready)

---

## 4. Domain-Specific Guides

Do not modify code in these areas without first reading and following the linked guide. If the guide conflicts with the requested change, **refuse the change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
