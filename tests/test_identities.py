"""The tests that matter: algebraic identities that must hold exactly."""
import numpy as np
import pytest
from aei_audit import indices as ix, aggregation as ag


def _fake(seed=0):
    rng = np.random.default_rng(seed)
    n = len(ag.CONTIGUOUS_48)
    pop = rng.lognormal(14.5, .9, n)
    conv = rng.poisson(pop / 1000 * rng.lognormal(0, .3, n)).astype(float) + 1
    return ag.CONTIGUOUS_48, conv, pop


@pytest.mark.parametrize("seed", range(5))
def test_theil_decomposition_is_exact(seed):
    states, conv, pop = _fake(seed)
    g = [ag.CENSUS_DIVISION[s] for s in states]
    d = ix.theil_decompose(conv / pop, g, weights=pop)
    assert abs(d["check_residual"]) < 1e-12


def test_clr_sums_to_zero():
    _, conv, _ = _fake()
    assert abs(ix.clr(ix.usage_share(conv)).sum()) < 1e-10


def test_aui_is_scale_invariant():
    """Multiplying all counts by a constant must not change the index."""
    states, conv, pop = _fake()
    a = ix.representation_ratio(conv, pop)
    b = ix.representation_ratio(conv * 137.0, pop)
    assert np.allclose(a, b)


def test_aui_weighted_mean_is_one():
    """Denominator-weighted mean of a representation ratio equals 1."""
    _, conv, pop = _fake()
    r = ix.representation_ratio(conv, pop)
    assert abs(np.sum(r * pop / pop.sum()) - 1.0) < 1e-10


def test_random_partitions_are_contiguous():
    rng = np.random.default_rng(1)
    for _ in range(20):
        lab = ag.random_contiguous_partition(9, rng)
        assert len(lab) == len(ag.CONTIGUOUS_48)
        for k in set(lab.values()):
            members = {s for s, v in lab.items() if v == k}
            seen, stack = set(), [next(iter(members))]
            while stack:
                s = stack.pop()
                if s in seen:
                    continue
                seen.add(s)
                stack += [n for n in ag.STATE_ADJACENCY[s]
                          if n in members and n not in seen]
            assert seen == members, "region is not contiguous"


def test_aggregation_recomputes_not_averages():
    """Aggregated AUI must come from summed counts, not averaged ratios."""
    import pandas as pd
    states, conv, pop = _fake()
    panel = pd.DataFrame({"state": states, "conversations": conv,
                          "denominator": pop})
    out = ag.aggregate_panel(panel, ag.CENSUS_REGION)
    assert abs(np.sum(out["aui"] * out["denominator"] / out["denominator"].sum()) - 1) < 1e-10
