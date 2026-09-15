# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The tutorial notebooks still run.

``nbsphinx_execute = 'never'``, so the published pages show whatever outputs were
committed. That is fast and reproducible, and it also means a notebook can rot
for months without anyone noticing: the page keeps rendering the old outputs.
These tests re-execute the notebooks that finish in reasonable time.

The re-execution tests are marked ``slow`` and deselected by default -- the
QuVINE example takes about seven minutes. Run them with ``pytest -m slow``. The
committed-output checks alongside them are cheap and always run.

Each notebook runs against a *copy* of its own directory, so a notebook that
writes results (the data-generation one does) cannot leave anything behind in
the checkout.
"""

from __future__ import annotations

import shutil

import pytest

from .conftest import REPO_ROOT, subprocess_env

# Both are declared in the [dev] extra and installed explicitly by the CI
# install-matrix legs, so they are always present wherever this file is
# collected. Guarding them meant every test below skipped silently in any
# environment that lacked either -- 21 tests reporting as "not run" is
# indistinguishable from "passed" in a green log.
import nbclient
import nbformat

# Timings measured on a laptop:
#   example_data_generation  ~20 s   (offline)
#   example_quvine           ~7.5 min (offline)
#   catboost_and_tabpfn      ~20 s   (needs the [tabpfn] extra and, on a machine with no
#                                     cached checkpoint, network access to download it)
#   example_qprofiler_v2     ~5 min  (needs the [tabpfn] extra; 300 tuned model fits)
# Notebooks needing anndata/scanpy or a real quantum backend are deliberately
# absent: they cannot run in a bare CI environment.
NOTEBOOKS = [
    "tutorial/Artificial_data_generation/example_data_generation.ipynb",
    "tutorial/QuVINE/example_quvine.ipynb",
    # Compares the two tuning engines and exercises the quantum tuning path, so it is
    # the end-to-end guard for both: a change that breaks either shows up here.
    "tutorial/Hyperparameter_Tuning/optuna_vs_gridsearch.ipynb",
    # The two learners added alongside XGBoost, and the only notebook exercising the
    # CatBoost bootstrap guard. It fits TabPFN for real rather than degrading, which is the
    # point -- the pinned v2 weights need no token -- so unlike the others it is NOT
    # runnable without the optional extra, and is skipped when that is absent. See
    # NOTEBOOKS_NEEDING_TABPFN below.
    "tutorial/CatBoost_and_TabPFN/catboost_and_tabpfn.ipynb",
    # The widest QProfiler configuration the suite executes: all ten models under Optuna
    # over 3 datasets x 5 splits x 2 embeddings. It is the end-to-end guard for the
    # ModelResults.csv writer, because it is the only notebook that runs tuned classical
    # models alongside an UNTUNED quantum one -- the combination that made the file
    # ragged and unreadable (see tests/test_model_results_csv.py). It fits TabPFN, so it
    # is listed in NOTEBOOKS_NEEDING_TABPFN below.
    "tutorial/QProfiler_v2/example_qprofiler_v2.ipynb",
]

#: Notebooks that fit TabPFN unconditionally, and so require the [tabpfn] extra.
#:
#: This mattered as soon as the notebook stopped degrading gracefully: before the v2 pin its
#: TabPFN cell caught ImportError and printed an explanation, so it executed with or without
#: the extra. Now it fits for real, and a bare-install run of `pytest -m slow` would *fail*
#: rather than skip. CI does not currently reach it -- the default `addopts` excludes `slow`
#: -- so nothing would have caught this until someone ran the slow tier by hand.
NOTEBOOKS_NEEDING_TABPFN = frozenset(
    {
        "tutorial/CatBoost_and_TabPFN/catboost_and_tabpfn.ipynb",
        "tutorial/QProfiler_v2/example_qprofiler_v2.ipynb",
    }
)


@pytest.mark.slow
@pytest.mark.parametrize("relative_path", NOTEBOOKS)
def test_the_notebook_executes(relative_path, tmp_path, monkeypatch, tabpfn_skip_reason):
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    if relative_path in NOTEBOOKS_NEEDING_TABPFN and tabpfn_skip_reason is not None:
        pytest.skip(tabpfn_skip_reason)

    notebook_path = REPO_ROOT / relative_path
    assert notebook_path.is_file(), f"{relative_path} is listed but missing"

    # Run against a copy: the data-generation notebook writes a data/ directory
    # next to itself, and a test must not mutate the checkout.
    sandbox = tmp_path / notebook_path.parent.name
    shutil.copytree(notebook_path.parent, sandbox)

    for key, value in subprocess_env().items():
        monkeypatch.setenv(key, value)

    notebook = nbformat.read(sandbox / notebook_path.name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(sandbox)}},
    )
    try:
        client.execute()
    except CellExecutionError as failure:
        pytest.fail(f"{relative_path} failed to execute:\n{failure}")


# One tree, not two. ``docs/source/tutorials/`` was a second copy of these same
# notebooks and is now generated from ``tutorial/`` by ``conf.py`` at build time,
# so it is absent on a clean checkout and a duplicate of every entry here after a
# build -- either way, nothing to execute.
ALL_NOTEBOOKS = sorted(
    str(path.relative_to(REPO_ROOT))
    for path in (REPO_ROOT / "tutorial").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in path.parts
)

# Notebooks whose committed outputs stop partway. Each would be listed with the
# reason it could not be completed, so an entry is a debt record rather than a
# permanent exemption -- delete it once the notebook is re-executed. Empty
# because every notebook in the tree is now either a clean template or fully
# executed; the mechanism is kept so a future truncation is recorded here
# deliberately rather than silently weakening the assertion below.
KNOWN_TRUNCATED: dict[str, str] = {}


def _executed_and_total(relative_path):
    """Count code cells that carry evidence of having run.

    Neither signal alone is reliable: a cell that imports a module runs and
    prints nothing, and several notebooks in this tree were saved with outputs
    intact but ``execution_count`` cleared. Either one counts as evidence.
    """
    notebook = nbformat.read(REPO_ROOT / relative_path, as_version=4)
    code_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.strip()
    ]
    executed = [
        cell
        for cell in code_cells
        if cell.execution_count is not None or cell.get("outputs")
    ]
    return len(executed), len(code_cells)


@pytest.mark.parametrize("relative_path", ALL_NOTEBOOKS)
def test_no_notebook_is_half_executed(request, relative_path):
    """A notebook is either a clean template or fully executed -- never in between.

    Half-executed is the state that misleads. ``nbsphinx_execute = 'never'``, so
    the published page renders the committed outputs and then simply stops: the
    reader sees a tutorial that appears to work right up to the cell where it
    was abandoned. A notebook with no outputs at all is honest by comparison,
    and is what several of these are by design.
    """
    if relative_path in KNOWN_TRUNCATED:
        request.node.add_marker(
            pytest.mark.xfail(reason=KNOWN_TRUNCATED[relative_path], strict=True)
        )
    executed, total = _executed_and_total(relative_path)
    assert executed in (0, total), (
        f"{relative_path}: {executed} of {total} code cells have outputs, so the "
        f"published page stops partway through the tutorial"
    )


def test_the_truncation_list_names_only_real_notebooks():
    """A stale entry would silently exempt a notebook that no longer exists."""
    missing = [path for path in KNOWN_TRUNCATED if path not in ALL_NOTEBOOKS]
    assert not missing, f"KNOWN_TRUNCATED names notebooks that are gone: {missing}"
