# DanteDojo

DanteDojo is the sovereign AI training, evaluation, specialization, and inference
organ for Real Empanada Corp. It provides a unified command-line interface and
Python library for the full model lifecycle: hardware detection, fine-tuning,
quantization, serving, evaluation, and promotion gating — all governed by a
constitutional substrate that enforces evidence emission, privacy-tier separation,
and deterministic replayability on every operation.

---

## Quick Install

```bash
pip install -e ".[dev]"
```

For training dependencies:

```bash
pip install -e ".[dev,train]"
```

---

## Key Commands

| Command | Description |
|---|---|
| `dojo detect` | Detect hardware, VRAM, compute capability, and recommend precision profile |
| `dojo train` | Launch a fine-tuning run (Unsloth/Axolotl backend) |
| `dojo idle-train` | Run fine-tuning only during idle windows with thermal/VRAM/power guards |
| `dojo serve` | Start a vLLM or llama.cpp inference server |
| `dojo eval` | Run lm-eval benchmarks against a checkpoint |
| `dojo promote` | Run the three-way promotion gate and issue a Constitutional Passport |

---

## Documentation

See [`docs/`](docs/) for architecture decisions, open questions, and subsystem
references.

Constitutional rules: [`CONSTITUTIONAL.md`](CONSTITUTIONAL.md)

Changelog: [`CHANGELOG.md`](CHANGELOG.md)
