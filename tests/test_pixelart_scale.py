import pytest
import imaging.pixelart_scale as mod

CASES = [
    ("integer_ratio", (1,)),
    ("threshold_local_contrast", (1,)),
    ("neighborhood_cross_tile", (1,)),
    ("global_palette", (1,)),
    ("pad_size", (1,)),
    ("edge_check", (1,)),
    ("load", (1,)),
    ("scale", (1,)),
    ("quantize", (1,)),
    ("export", (1,)),
    ("validate_scale", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
