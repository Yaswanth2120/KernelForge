from ._common import require_gpu


def causal_attention(q, k, v, sm_scale=None, return_lse=False):
    """Public interface reserved for the upcoming kernel implementation."""
    require_gpu()
    raise NotImplementedError("causal_attention kernel is not implemented yet.")
