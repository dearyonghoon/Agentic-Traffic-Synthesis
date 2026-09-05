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

## Dataset Sources

Raw traffic data are **not redistributed in this repository**.

- **GAViST5G**: Kaggle dataset `ahassanein/aggregated-gaming-and-video-streaming-traffic-for-5g`.
- **CESNET-TimeSeries24**: Zenodo record `13382427`, using `ip_addresses_sample.tar.gz`. The frozen archive MD5 is `08451dab2d1eddb29a79467b03289232`.
- **MAWI**: samplepoint-F 2024 traces at 14:00 on Jan. 1, Mar. 1, May 1, Jul. 1, Sep. 1, and Nov. 1. Jan/Mar/May/Jul are training traces, Sep is validation, and Nov is test.

Please follow the original dataset licenses and terms of use.

## Dataset Preparation

By default, the scripts use:

```text
/data/dataset/AgenticTraffic/
├── gavist5g/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── cesnet_ts24/
│   ├── raw/
│   ├── processed/
│   └── metadata/
└── mawi/
    ├── raw/
    ├── processed/
    └── metadata/
```

A different root can be supplied with `--data-root`.

Prepare GAViST5G:

```bash
python scripts/prepare_gavist5g.py
```

Prepare CESNET-TimeSeries24:

```bash
python scripts/prepare_cesnet.py
```

Prepare MAWI:

```bash
python scripts/prepare_mawi.py
```

MAWI preprocessing requires the system executable `tshark`. On Debian/Ubuntu it can be installed with the system package manager.

The final frozen representations are:

- GAViST5G: non-overlapping 60-s windows, application-stratified source-file split, and train-only semantic thresholds.
- CESNET-TimeSeries24: strict non-overlapping 12-row windows with consecutive integer time indices, source-file group split, and train-only semantic thresholds.
- MAWI: 30-s bytes/packet sequences with 5-s stride and train-only semantic thresholds.

Audit the resulting files:

```bash
python scripts/verify_dataset_setup.py --strict-counts
```

The strict audit checks the frozen GAViST5G and CESNET counts and the MAWI trace/split/sequence protocol.


## Frozen Original-40 Reproduction

The repository includes the frozen unsupported-intent benchmark and the
pretrained Residual FiLM-CFM checkpoints required for the CESNET-TimeSeries24
and GAViST5G original-40 audit.

Verify the public checkpoints:

    sha256sum -c pretrained/SHA256SUMS

After preparing the datasets, reproduce the frozen benchmark:

    python scripts/verify_public_original40.py

To use a non-default dataset location:

    python scripts/verify_public_original40.py --data-root /path/to/AgenticTraffic

The frozen reference contains 40 unsupported intents per dataset. With an
attempt budget of 8, the reference aggregate success counts are:

| Dataset | Copula | Residual FiLM-CFM |
|---|---:|---:|
| CESNET-TimeSeries24 | 5 / 40 | 11 / 40 |
| GAViST5G | 6 / 40 | 4 / 40 |

The reproduction script checks equality of these paper-level aggregate
outcomes and also reports per-intent exact fidelity. The aggregate results are
the default reproduction criterion. For a stricter diagnostic, run:

    python scripts/verify_public_original40.py --require-case-exact

## Repository Structure

```text
agent/          Agentic control logic
generators/     Traffic synthesis tools
configs/        Frozen experiment and domain configurations
benchmarks/     Intent benchmarks used in the paper
scripts/        Dataset, evaluation, and reproduction scripts
notebooks/      Supplementary analyses
results/        Paper tables and figures
examples/       End-to-end and control-plane usage examples
tests/          Public behavioral tests
docs/           Implementation notes
```

## Code Release Status

Core agent and synthesis modules, dataset-preparation scripts, behavioral tests, the frozen original-40 unsupported-intent benchmark, and the CESNET/GAViST Residual FiLM-CFM checkpoints are publicly available. The repository includes a standalone reproduction script for the frozen vector-domain audit. Additional paper-table and end-to-end evaluation scripts will be added as the release is finalized.

## License

MIT License.
