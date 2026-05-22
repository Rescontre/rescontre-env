"""Netting invariants, ported from the Rust core's test suite (src/netting.rs).

These guard the property the whole environment leans on: settlement preserves
conservation and never invents money.
"""

from rescontre_env import netting


def graph_from(edges):
    g: netting.Graph = {}
    for frm, to, amt in edges:
        g.setdefault(frm, {})
        g.setdefault(to, {})
        g[frm][to] = amt
    return g


def assert_invariants(label, graph):
    transfers = netting.compute_net_transfers(graph)
    assert netting.conservation_holds(graph, transfers), f"{label}: conservation"
    for t in transfers:
        assert t.amount > 0, f"{label}: non-positive transfer"
        assert t.frm != t.to, f"{label}: self-loop"
    # Optimal transfer count bound: at most max(0, active_nodes - 1).
    active = len(netting.net_positions(graph))
    assert len(transfers) <= max(0, active - 1), f"{label}: transfer-count bound"
    return transfers


def test_three_cycle_cancels_completely():
    g = graph_from([("A", "B", 100), ("B", "C", 100), ("C", "A", 100)])
    assert netting.compute_net_transfers(g) == []


def test_chain_reroutes():
    g = graph_from([("A", "B", 50), ("B", "C", 50)])
    t = assert_invariants("chain", g)
    assert len(t) == 1
    assert (t[0].frm, t[0].to, t[0].amount) == ("A", "C", 50)


def test_partial_cycle_residual():
    g = graph_from([("A", "B", 100), ("B", "C", 100), ("C", "A", 60)])
    t = assert_invariants("partial-cycle", g)
    assert len(t) == 1
    assert (t[0].frm, t[0].to, t[0].amount) == ("A", "C", 40)


def test_star_payout():
    g = graph_from([("A", "B", 10), ("A", "C", 20), ("A", "D", 30), ("A", "E", 40)])
    t = assert_invariants("star", g)
    assert len(t) == 4


def test_disjoint_pairs():
    g = graph_from([("A", "B", 100), ("B", "A", 100), ("C", "D", 25)])
    t = assert_invariants("disjoint", g)
    assert len(t) == 1
    assert (t[0].frm, t[0].to) == ("C", "D")


def test_determinism():
    g = graph_from([("A", "B", 30), ("B", "C", 50), ("C", "D", 20),
                    ("D", "A", 10), ("A", "C", 5)])
    first = netting.compute_net_transfers(g)
    for _ in range(16):
        assert netting.compute_net_transfers(g) == first


def test_empty_graph():
    assert netting.compute_net_transfers({}) == []


def test_compression_ratio_cycle():
    # A full 3-cycle has gross volume 300 but settles to 0 -> defined as 1.0.
    g = graph_from([("A", "B", 100), ("B", "C", 100), ("C", "A", 100)])
    t = netting.compute_net_transfers(g)
    assert netting.gross_volume(g) == 300
    assert netting.settled_volume(t) == 0
    assert netting.compression_ratio(g, t) == 1.0
