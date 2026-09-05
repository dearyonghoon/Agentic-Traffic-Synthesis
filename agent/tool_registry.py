from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple


# Frozen final profile orders used in the paper's final execution path.
# CESNET/GAViST order is the final Exp.18 profile order.
# MAWI order is the hard-consistent 10d profile order.
DEFAULT_TOOL_ORDER = {
    "cesnet_ts24": ("cfm", "copula"),
    "gavist5g": ("copula", "cfm"),
    "mawi": ("scfm", "copula"),
}


@dataclass
class FrozenToolRegistry:
    """
    Persistent empirical tool profile used only to choose execution order.

    The registry does not change the requested traffic intent.
    """
    orders: Mapping[str, Sequence[str]] = None

    def __post_init__(self):
        if self.orders is None:
            self.orders = DEFAULT_TOOL_ORDER
        self.orders = {
            str(ds): tuple(order)
            for ds, order in self.orders.items()
        }

    def get_order(self, dataset: str) -> Tuple[str, ...]:
        if dataset not in self.orders:
            raise KeyError(f"No frozen tool profile for dataset={dataset!r}")
        return tuple(self.orders[dataset])
