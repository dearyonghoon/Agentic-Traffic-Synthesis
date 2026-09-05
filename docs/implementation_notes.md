# Public Core v1

This bundle modularizes the frozen implementation used by the paper:

- 05b: Qwen2.5-7B-Instruct intent compiler and conflict handling
- 07: CESNET/GAViST copula synthesis and verifier construction
- 08b: CESNET/GAViST Residual-FiLM CFM
- 09 / 18: final empirical tool order for CESNET/GAViST
- 10c / 10d: MAWI temporal verifier, Semantic-Consistency CFM, and hard feasibility guard

## Important behavioral invariant

Automatic fallback changes the **tool**, not the **intent**.

The earlier semantic-revision baseline (`single_probe_revision` /
`run_capability_router_replan`) is intentionally **not** included in the
proposed-agent implementation.

## Natural-language scope

The frozen 05b natural-language compiler supports:
- `gavist5g`
- `cesnet_ts24`

MAWI is supported by the public agent as a structured `TrafficIntent`. This
avoids silently extending the frozen compiler beyond the evaluated setup.

## Checkpoints

Large model checkpoints are not included in this bundle. The generator classes
expect checkpoint paths and preserve the frozen inference architectures.

## Exact MAWI feasibility thresholds

Use the exact `semantic_thresholds_log1p` from
`10c_mawi_metadata_v1.json` with:

```python
guard = FeasibilityGuard.from_mawi_metadata("10c_mawi_metadata_v1.json")
```

This avoids rounding the threshold values in the public implementation.
