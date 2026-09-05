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

## Repository Structure

```text
agent/          Agentic control logic
generators/     Traffic synthesis tools
configs/        Frozen experiment and domain configurations
benchmarks/     Intent benchmarks used in the paper
scripts/        Evaluation and reproduction scripts
notebooks/      Supplementary analyses
results/        Paper tables and figures
examples/       End-to-end usage examples
```

## Code Release Status

This repository accompanies the paper and is being organized for reproducible public release.

## License

MIT License.
