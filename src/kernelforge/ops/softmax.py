from ._common import require_gpu


def softmax(x, dim=-1):
    """Public interface reserved for the upcoming kernel implementation."""
    require_gpu()
    raise NotImplementedError("softmax kernel is not implemented yet.")
