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
