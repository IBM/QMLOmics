from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {relative_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_package(package_name: str, relative_path: str):
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO_ROOT / relative_path)]
        sys.modules[package_name] = package
    return package


# ----------------------------------------------------------------------------------
# TabPFN: installed, and able to reach its weights, are two different questions
# ----------------------------------------------------------------------------------
#
# `tabpfn` is an optional extra ([tabpfn]). QBioCode's pinned model version (v2) needs
# no token and no license acceptance, so with the extra installed a fit normally just
# works -- but a test cannot assume it: the extra may be absent, the first fit needs
# network access to download the checkpoint, and a config that opts into v2.5/v2.6/v3
# hits a license gate. So a test that fits TabPFN has three possible states, not two:
# the extra is missing, the extra is present but the weights are not, or it works. Only
# the third can assert on a score; the other two must skip rather than fail, or CI goes
# red for a reason that is nobody's bug.
#
# The probe runs in a subprocess for two reasons: `import tabpfn` imports torch, which
# tests/test_openmp_import_order.py exists to keep out of this process, and a real fit
# is the only authoritative answer about the weights -- inspecting the cache directory
# would be guessing at upstream's internals.

_TABPFN_PROBE = textwrap.dedent(
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import qbiocode  # orders the OpenMP runtimes before torch arrives
    from qbiocode.learning.compute_tabpfn import tabpfn_is_available

    if not tabpfn_is_available():
        print("MISSING: the [tabpfn] extra is not installed")
        raise SystemExit(0)

    # Through compute_tabpfn, not a bare TabPFNClassifier: the bare constructor defaults to
    # the v3 checkpoint, which is license-gated, so probing that way reported "gated" even
    # after QBioCode pinned the ungated v2 -- and every fitting test skipped for a reason
    # that no longer applied.
    from qbiocode.learning.compute_tabpfn import compute_tabpfn

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    try:
        compute_tabpfn(
            X[:28], X[28:], y[:28], y[28:], {"grid_search": False},
            n_estimators=1, device="cpu", random_state=0,
        )
    except Exception as exc:
        print(f"GATED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    else:
        print("READY")
    """
)


def _probe_tabpfn():
    """Return ``None`` when TabPFN can fit, or a skip reason explaining why not."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _TABPFN_PROBE],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return "TabPFN probe timed out (weight download or import too slow)"
    out = completed.stdout.strip().splitlines()
    verdict = out[-1] if out else ""
    if verdict == "READY":
        return None
    if verdict.startswith(("MISSING:", "GATED:")):
        return verdict
    return (
        f"TabPFN probe was inconclusive (exit {completed.returncode}): "
        f"{completed.stderr.strip()[-300:] or 'no output'}"
    )


@pytest.fixture(scope="session")
def tabpfn_skip_reason():
    """``None`` when TabPFN is usable, else the reason to skip -- probed once."""
    return _probe_tabpfn()


@pytest.fixture(scope="session")
def tabpfn_ready(tabpfn_skip_reason):
    """Skip the requesting test unless TabPFN is installed *and* its weights load."""
    if tabpfn_skip_reason is not None:
        pytest.skip(tabpfn_skip_reason)
    return True
