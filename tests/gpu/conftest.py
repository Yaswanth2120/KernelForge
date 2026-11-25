"""Persist owner-run autotuning evidence without importing a GPU backend during collection."""
import json
import sys
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """Save configuration diagnostics only if a real softmax backend was loaded."""
    backend = sys.modules.get('kernelforge.ops._softmax_kernels')
    if backend is None:
        return
    directory = Path(session.config.rootpath) / 'reports' / 'raw'
    directory.mkdir(parents=True, exist_ok=True)
    artifact = {'kind': 'kernelforge.softmax.autotune.v1', 'pytest_exit_code': int(exitstatus),
                'configurations': list(backend.PRUNING_LOG.values())}
    (directory / 'softmax_autotune.json').write_text(json.dumps(artifact, indent=2) + '\n')
