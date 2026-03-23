"""Shared test setup: mock out heavy transitive dependencies."""

import sys
import types

def _fake(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod


def setup_mocks():
    """Call once before importing proteina-complexa modules."""
    if "torch_scatter" in sys.modules:
        return  # already set up

    sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
    sys.modules["torch_scatter"].scatter_mean = None

    # Paths
    sys.path.insert(0, "proteina-complexa/community_models")
    sys.path.insert(0, "proteina-complexa/src")

    # Real openfold.np (needs dm-tree, numpy)
    import openfold
    import openfold.np.residue_constants

    # Fake openfold.model (needs scipy, ml_collections — too heavy)
    import torch

    class _F(torch.nn.Module):
        def __init__(self, *a, **kw): super().__init__()
        def forward(self, x, *a, **kw): return x

    _fake("openfold.model")
    for n, a in [
        ("openfold.model.dropout", {"DropoutColumnwise": _F, "DropoutRowwise": _F}),
        ("openfold.model.pair_transition", {"PairTransition": _F}),
        ("openfold.model.triangular_attention", {"TriangleAttentionStartingNode": _F, "TriangleAttentionEndingNode": _F}),
        ("openfold.model.triangular_multiplicative_update", {"TriangleMultiplicationIncoming": _F, "TriangleMultiplicationOutgoing": _F}),
    ]:
        _fake(n, a)
