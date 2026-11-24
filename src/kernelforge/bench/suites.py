from itertools import product

ROW_WIDTHS = (2048, 4096, 8192)
ROW_DTYPES = {"fp16": 2, "bf16": 2, "fp32": 4}
ATTENTION_PAIRS = ((8, 2048), (4, 4096), (2, 8192))
MIN_INPUT_BYTES = 512 * 1024**2
PREREG_SHA256 = "940b2749bf6a8214d5e6282dfbbf0e6e687aa4ed1cc2d5d4e94ea789fd1ead7e"


def selected_rows():
    """Use the smallest row count meeting the registered input size."""
    return [{"M": (MIN_INPUT_BYTES + n * size - 1) // (n * size),
             "N": n, "dtype": dtype}
            for n, (dtype, size) in product(ROW_WIDTHS, ROW_DTYPES.items())]


def selected_attention():
    """Enumerate the frozen attention Cartesian product."""
    return [{"B": b, "H": 32, "S": s, "D": d, "dtype": dtype}
            for (b, s), d, dtype in product(ATTENTION_PAIRS, (64, 128), ("fp16", "bf16"))]


# Full sweep (reported, not pass/fail): small, path-threshold, large, and irregular widths.
SWEEP_WIDTHS = (256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1000, 4097, 31337)
COPY_BUFFER_BYTES = 1024**3


def rows_for(n, elem_size):
    """Pick the smallest row count whose input reaches the registered 512 MiB."""
    return (MIN_INPUT_BYTES + n * elem_size - 1) // (n * elem_size)


def row_sweep():
    """Row-op (softmax, LayerNorm) sweep cases; the pre-registered pass/fail cases are flagged as selected."""
    selected = {(case["N"], case["dtype"]) for case in selected_rows()}
    return [{"M": rows_for(n, size), "N": n, "dtype": dtype, "selected": (n, dtype) in selected}
            for n, (dtype, size) in product(SWEEP_WIDTHS, ROW_DTYPES.items())]


def copy_cases():
    """One >= 1 GiB buffer per copy implementation, counted as one read plus one write."""
    return [{"buffer_bytes": COPY_BUFFER_BYTES, "dtype": "fp32"}]


def check_prereg(path):
    """Reject edits to the registered protocol or executable suite definitions."""
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != PREREG_SHA256:
        raise ValueError("PREREG.md changed; review and document the original and new protocol.")
    if (ROW_WIDTHS != (2048, 4096, 8192)
            or ROW_DTYPES != {"fp16": 2, "bf16": 2, "fp32": 4}
            or ATTENTION_PAIRS != ((8, 2048), (4, 4096), (2, 8192))
            or MIN_INPUT_BYTES != 512 * 1024**2):
        raise ValueError("Suite definitions differ from PREREG.md")


# Attention sweep (reported, not pass/fail): B * S fixed at 16384 tokens, H = 32, plus irregular S.
ATTENTION_SWEEP_S = (512, 1024, 2048, 4096, 8192, 16384)
ATTENTION_IRREGULAR = ((16, 1000), (4, 4097))
ATTENTION_MEMORY_S = (1024, 2048, 4096, 8192, 16384)


def attention_sweep():
    """Sweep cases; the pre-registered pass/fail cases are flagged as selected."""
    selected = [{**case, "selected": True} for case in selected_attention()]
    pairs = [(16384 // s, s) for s in ATTENTION_SWEEP_S] + list(ATTENTION_IRREGULAR)
    extra = [{"B": b, "H": 32, "S": s, "D": d, "dtype": dtype, "selected": False}
             for (b, s), d, dtype in product(pairs, (64, 128), ("fp16", "bf16"))
             if (b, s) not in ATTENTION_PAIRS]
    return selected + extra


def attention_memory_cases():
    """Peak-memory scaling in S at B=1, H=32, D=64, fp16 (math backend shows O(S^2))."""
    return [{"B": 1, "H": 32, "S": s, "D": 64, "dtype": "fp16"} for s in ATTENTION_MEMORY_S]
