# Agentic Traffic Synthesis

Official implementation of:

**From Language to Traffic: Agentic AI for Intent-Driven Network Traffic Synthesis**

> A generator produces traffic, whereas the agent decides whether and how traffic generation should be carried out.

## Overview

This repository implements an intent-driven agentic framework for network traffic synthesis.

The agent performs:

1. natural-language intent compilation,
2. clarification and structural feasibility checking,
3. empirical traffic-generator selection,
4. traffic synthesis,
5. output verification, and
6. intent-preserving fallback when verification fails.

A failed synthesis attempt changes the **tool**, not the **user intent**.

## Quick Start

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the lightweight control-plane demo:

```bash
python examples/control_plane_demo.py
```

The demo does not require traffic datasets or pretrained checkpoints. It verifies the key execution invariant of the proposed agent: after a verifier failure, the agent may switch synthesis tools, but every tool receives the same original traffic intent.

Run the public tests:

```bash
python -m pytest -q
```

The tests include intent-preserving fallback and the frozen MAWI structural-feasibility guard.

## Repository Structure

```text
agent/          Agentic control logic
generators/     Traffic synthesis tools
configs/        Frozen experiment and domain configurations
benchmarks/     Intent benchmarks used in the paper
scripts/        Evaluation and reproduction scripts
notebooks/      Supplementary analyses
results/        Paper tables and figures
examples/       End-to-end and control-plane usage examples
tests/          Public behavioral tests
docs/           Implementation notes
```

## Code Release Status

Core agent and synthesis modules are now available. Dataset preparation, pretrained checkpoints, and full paper-reproduction scripts will be added as the public release is finalized.

## License

MIT License.
