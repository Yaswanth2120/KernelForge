from ._common import require_gpu


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, return_stats=False):
    """Public interface reserved for the upcoming kernel implementation."""
    require_gpu()
    raise NotImplementedError("layer_norm kernel is not implemented yet.")
