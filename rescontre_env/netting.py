"""Multilateral netting + conservation check.

This is a faithful Python port of the Rescontre facilitator's production netting
core (`src/netting.rs`). It is the *deterministic* heart of the environment: given
a graph of bilateral obligations, it produces the provably minimum-transfer,
minimum-volume set of net transfers, and certifies that conservation holds.

Two reasons it lives here unchanged in spirit:

1. It is the env's settlement dynamics. "Given these obligations, what does it
   cost to clear them?" is answered by netting, not by a made-up cost model.
2. Its conservation invariant is the env's *reward-validity oracle*. A policy
   that appears to settle cheaply while violating conservation has cheated; the
   reward is gated on `conservation_holds()` returning True. Reward integrity is
   not a soft signal here, it is a checkable property of the books.

Amounts are integers (minor units, e.g. micro-USD) exactly as in production, so
there is no floating-point drift in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

# A bilateral obligation graph: graph[debtor][creditor] = amount owed (positive).
Graph = dict[str, dict[str, int]]


@dataclass(frozen=True)
class NetTransfer:
    """A single net transfer produced by debtor-creditor matching."""

    frm: str
    to: str
    amount: int


def net_positions(graph: Graph) -> dict[str, int]:
    """Per-node net position: (incoming - outgoing).

    Positive = net creditor (is owed), negative = net debtor (owes). By
    conservation these always sum to zero across all nodes.
    """
    net: dict[str, int] = {}
    for frm, neighbors in graph.items():
        for to, weight in neighbors.items():
            if weight > 0:
                net[frm] = net.get(frm, 0) - weight
                net[to] = net.get(to, 0) + weight
    return {node: bal for node, bal in net.items() if bal != 0}


def compute_net_transfers(graph: Graph) -> list[NetTransfer]:
    """Optimal debtor-creditor matching.

    After computing each node's net position, we ignore the original edge
    topology entirely and match debtors to creditors with a two-pointer sweep.
    This yields the minimum number of transfers and minimum total volume.
    O(V log V): it is just a sort. Deterministic (ties broken by name).
    """
    net = net_positions(graph)

    debtors = [(node, -bal) for node, bal in net.items() if bal < 0]
    creditors = [(node, bal) for node, bal in net.items() if bal > 0]

    # Descending by amount, then ascending by name, exactly like production.
    debtors.sort(key=lambda x: (-x[1], x[0]))
    creditors.sort(key=lambda x: (-x[1], x[0]))

    transfers: list[NetTransfer] = []
    di = ci = 0
    # Mutable amounts as we draw down each side.
    damt = [a for _, a in debtors]
    camt = [a for _, a in creditors]

    while di < len(debtors) and ci < len(creditors):
        amount = min(damt[di], camt[ci])
        transfers.append(NetTransfer(debtors[di][0], creditors[ci][0], amount))
        damt[di] -= amount
        camt[ci] -= amount
        if damt[di] == 0:
            di += 1
        if camt[ci] == 0:
            ci += 1

    return transfers


def _net_from_transfers(transfers: list[NetTransfer]) -> dict[str, int]:
    net: dict[str, int] = {}
    for t in transfers:
        net[t.frm] = net.get(t.frm, 0) - t.amount
        net[t.to] = net.get(t.to, 0) + t.amount
    return {node: bal for node, bal in net.items() if bal != 0}


def conservation_holds(graph: Graph, transfers: list[NetTransfer]) -> bool:
    """The reward-validity oracle.

    Returns True iff the transfer set preserves every node's net position and
    every transfer is strictly positive with no self-loops. This is the formal
    "no reward hacking" certificate: the books balance, or they do not.
    """
    if _net_from_transfers(transfers) != net_positions(graph):
        return False
    for t in transfers:
        if t.amount <= 0 or t.frm == t.to:
            return False
    return True


def gross_volume(graph: Graph) -> int:
    """Total obligation volume before netting (sum of all positive edges)."""
    return sum(w for nb in graph.values() for w in nb.values() if w > 0)


def settled_volume(transfers: list[NetTransfer]) -> int:
    """Total volume actually moved after netting compression."""
    return sum(t.amount for t in transfers)


def compression_ratio(graph: Graph, transfers: list[NetTransfer]) -> float:
    """gross / settled. Higher means netting saved more. 1.0 means no saving.

    This is the efficiency metric a settlement-timing policy is trying to
    maximize: defer and batch obligations so more of them cancel out.
    """
    gross = gross_volume(graph)
    settled = settled_volume(transfers)
    return gross / settled if settled > 0 else 1.0
