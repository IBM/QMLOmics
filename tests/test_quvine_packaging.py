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

"""
Guards on how the QuVINE app is packaged and how it fails when its extra is absent.

QuVINE's Python modules ship with QBioCode, but its third-party dependencies sit
behind the ``[quvine]`` extra. Three things must therefore hold, and each broke at
least once while the app was being ported:

1. ``import qbiocode`` must not need the extra, and neither must the method-name
   resolution that ``qbiocode.embeddings`` uses to decide where to route a name.
2. Every directory under ``qbiocode/apps/quvine/`` must contain ``__init__.py``,
   or ``[tool.setuptools.packages.find]`` silently drops it from the wheel while a
   source checkout keeps working.
3. Names that ``embed()`` accepts must also resolve and be listed. ``quvine_fused``
   was accepted by ``embed()`` but unknown to ``resolve_method``/``list_methods``,
   so the ``quvine`` CLI rejected its own default method.

These tests import only the stdlib plus the dependency-free corners of the app, so
they run on a bare install.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QUVINE_ROOT = REPO_ROOT / "qbiocode" / "apps" / "quvine"

sys.path.insert(0, str(REPO_ROOT))

from qbiocode.apps.quvine import _deps  # noqa: E402
from qbiocode.apps.quvine.api import aliases  # noqa: E402


def _read_requirements(name):
    """Package names from a requirements file, comments and ``-r`` lines dropped."""
    lines = (REPO_ROOT / "requirements" / name).read_text().splitlines()
    specs = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            specs.append(line)
    return specs


def _distribution_of(spec):
    for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
        spec = spec.split(sep, 1)[0]
    return spec.strip().lower().replace("_", "-")


# ---------------------------------------------------------------------------
# 1. The extra is genuinely optional
# ---------------------------------------------------------------------------

def test_import_qbiocode_and_resolve_methods_without_the_quvine_extra():
    """Blocking every [quvine] dependency must not break the package.

    Run in a subprocess with a meta-path finder that refuses the extra's modules,
    so the test result is the same whether or not the extra is installed locally.
    """
    blocked = sorted(_deps.OPTIONAL_DEPENDENCIES)
    program = f"""
import sys

BLOCKED = {blocked!r}


class Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, Blocker())

import qbiocode

assert callable(qbiocode.get_embeddings)
assert callable(qbiocode.evaluate_graph)
assert callable(qbiocode.scale_train_test)

# The routing probe qbiocode.embeddings relies on must work with the extra absent.
from qbiocode.apps.quvine import list_methods, resolve_method

assert resolve_method("quvine_rwr") == ("sgns", "rwr")
assert resolve_method("node2vec") == ("registry", "node2vec")
assert resolve_method("quvine_fused") == ("fused", "fused")
assert len(list_methods()) > 50

# Nothing above may have pulled a blocked module in.
leaked = sorted(m for m in BLOCKED if m in sys.modules and sys.modules[m] is not None)
assert not leaked, f"blocked modules were imported anyway: {{leaked}}"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, (
        "importing qbiocode required a [quvine] dependency:\n" + result.stderr
    )
    assert "OK" in result.stdout


def test_requesting_a_quvine_method_names_the_extra_and_the_install_command():
    message = str(
        _deps.build_message(
            "hiperwalk",
            method="quvine_ctqw",
            cause=ModuleNotFoundError("No module named 'hiperwalk'"),
        )
    )
    assert "quvine_ctqw" in message
    assert "'quvine' extra" in message
    assert 'pip install "qbiocode[quvine]"' in message
    assert "hiperwalk" in message


def test_a_dependency_error_is_still_an_import_error():
    """Callers that probe with ``except ImportError`` must keep working."""
    assert issubclass(_deps.QuvineDependencyError, ImportError)


# ---------------------------------------------------------------------------
# 2. Everything the app needs is declared and shipped
# ---------------------------------------------------------------------------

def test_optional_dependency_table_matches_the_quvine_requirements_file():
    """``_deps`` is what users read in an error message; keep it truthful."""
    declared = {_distribution_of(spec) for spec in _read_requirements("requirements-quvine.txt")}
    described = {
        _distribution_of(spec) for spec, _reason in _deps.OPTIONAL_DEPENDENCIES.values()
    }
    # setuptools is a transitive pin for node2vec's use of pkg_resources, not a
    # module QuVINE imports, so it is deliberately undescribed.
    assert described == declared - {"setuptools"}


def test_every_third_party_import_under_qbiocode_is_declared():
    """A dependency imported but undeclared only resolves by luck."""
    # Import name -> distribution name, for the cases where they differ.
    distribution_of_import = {
        "community": "python-louvain",
        "hydra": "hydra-core",
        "skdim": "scikit-dimension",
        "sklearn": "scikit-learn",
        "umap": "umap-learn",
        "yaml": "pyyaml",
        "igraph": "igraph",
        "PIL": "pillow",
    }
    stdlib = set(sys.stdlib_module_names)
    roots = set()
    for path in (REPO_ROOT / "qbiocode").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
            # Dependencies resolved through require_module("x") are invisible to an
            # import scan, so pick up their string arguments too.
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require_module":
                if node.args and isinstance(node.args[0], ast.Constant):
                    roots.add(str(node.args[0].value).split(".")[0])

    # Every tier that ships a runtime dependency, not just base and quvine. `tabpfn` is
    # declared in requirements-tabpfn.txt, and was invisible to this scan until
    # `qbiocode.utils.tabpfn_account` imported it with a real `from tabpfn... import`:
    # `compute_tabpfn` reaches it through `importlib.import_module("tabpfn")`, a string the
    # AST walk below cannot see. So this list must grow whenever a new optional tier is
    # added, or the guard silently stops covering it.
    declared = {
        _distribution_of(spec)
        for name in (
            "requirements-base.txt",
            "requirements-quvine.txt",
            "requirements-tabpfn.txt",
        )
        for spec in _read_requirements(name)
    }
    undeclared = sorted(
        root
        for root in roots
        if root not in stdlib
        and root != "qbiocode"
        and distribution_of_import.get(root, root).lower().replace("_", "-") not in declared
    )
    assert not undeclared, f"imported by qbiocode/ but declared nowhere: {undeclared}"


def test_every_quvine_directory_is_a_package():
    """Without ``__init__.py`` a directory is dropped from the built wheel.

    ``walks``, ``corpus``, ``utils`` and ``configs`` shipped without one, so the
    wheel was missing modules that a source checkout imported fine.
    """
    missing = sorted(
        str(d.relative_to(REPO_ROOT))
        for d in QUVINE_ROOT.rglob("*")
        if d.is_dir()
        and d.name != "__pycache__"
        and any(f.suffix == ".py" for f in d.iterdir())
        and not (d / "__init__.py").exists()
    )
    assert not missing, f"directories with modules but no __init__.py: {missing}"


def test_pyproject_declares_the_quvine_console_script_and_packaged_config():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'quvine = "qbiocode.apps.quvine.cli:main"' in pyproject
    assert '"qbiocode.apps.quvine" = ["configs/*.yaml"]' in pyproject
    assert (QUVINE_ROOT / "configs" / "config.yaml").exists()
    assert (QUVINE_ROOT / "cli.py").exists()


@pytest.mark.parametrize(
    "config",
    sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "qbiocode").rglob("configs/*.yaml")
    ),
)
def test_no_shipped_config_names_an_absolute_path(config):
    """A packaged default must resolve on the machine that installed it.

    ``qbiocode/apps/quvine/configs/config.yaml`` shipped in the wheel pointing
    ``data_path`` and ``runtime.output_dir`` at ``/dccstor/boseukb/...`` -- an
    internal research filesystem. It is the config the ``quvine`` CLI falls back
    to when ``--config`` is omitted, so the default could not resolve for anyone
    outside that cluster.

    Parametrized over every shipped config rather than asserting on the one that
    was wrong, because the next one will be added by copying this one.
    """
    text = (REPO_ROOT / config).read_text(encoding="utf-8")
    offenders = []
    for number, line in enumerate(text.splitlines(), start=1):
        code = line.split("#", 1)[0]
        if ":" not in code:
            continue
        value = code.split(":", 1)[1].strip().strip("'\"")
        # A leading slash in a *value* is an absolute POSIX path; "C:\\..." is the
        # Windows equivalent. Interpolations (${...}) and comments are not values.
        if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            offenders.append(f"{config}:{number}: {line.strip()}")
    assert not offenders, (
        "absolute filesystem paths in a config that ships inside the wheel, so "
        "the default cannot resolve on an installed copy:\n  "
        + "\n  ".join(offenders)
    )


def test_torch_is_a_base_dependency_not_a_quvine_extra():
    """``qbiocode.embeddings`` imports torch eagerly, so it cannot be optional.

    If torch were only in the extra, a bare install would fail at
    ``import qbiocode`` and the [quvine] install hint would be wrong advice.
    """
    base = {_distribution_of(s) for s in _read_requirements("requirements-base.txt")}
    extra = {_distribution_of(s) for s in _read_requirements("requirements-quvine.txt")}
    assert "torch" in base
    assert "torch" not in extra
    assert "torch" not in _deps.OPTIONAL_DEPENDENCIES


# ---------------------------------------------------------------------------
# 3. Resolvable names and dispatchable names are the same set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(aliases.FUSED_ALIASES))
def test_fused_names_resolve_and_are_listed(name):
    assert aliases.resolve_method(name) == ("fused", "fused")
    assert name in aliases.list_methods()
    assert name in aliases.list_methods("fused")


def test_the_cli_default_method_is_one_the_cli_accepts():
    """``--method`` defaulted to ``quvine_fused``, which validation then rejected."""
    tree = ast.parse((QUVINE_ROOT / "cli.py").read_text())
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                defaults[flags[0]] = kw.value.value
    assert defaults.get("--method"), "cli.py no longer declares a --method default"
    aliases.resolve_method(defaults["--method"])  # raises KeyError if unknown


def test_embed_dispatches_exactly_the_kinds_resolve_method_can_return():
    """A new resolve_method kind with no dispatch branch would fall through."""
    kinds = {aliases.resolve_method(n)[0] for n in aliases.list_methods()}
    assert kinds == {"sgns", "registry", "fused"}


def test_every_listed_method_resolves():
    unresolvable = []
    for name in aliases.list_methods():
        try:
            aliases.resolve_method(name)
        except KeyError:
            unresolvable.append(name)
    assert not unresolvable, f"listed but unresolvable: {unresolvable}"


def test_unknown_method_message_suggests_close_names():
    with pytest.raises(KeyError) as excinfo:
        aliases.resolve_method("quvine_fuzed")
    message = excinfo.value.args[0]
    assert "quvine_fuzed" in message
    assert "quvine_fused" in message


def test_documented_method_count_matches_the_code():
    """The docs quote a method count; keep it from going stale silently."""
    total = len(aliases.list_methods())
    for relative in ("docs/source/apps.rst", "docs/source/apps/quvine.rst"):
        text = (REPO_ROOT / relative).read_text()
        assert f"{total} named methods" in text or f"{total} method names" in text or \
            f"({total} in total)" in text, f"{relative} does not quote {total} methods"


class TestTheEdgeListLoaderDropsAHeaderRow:
    """A header row must not become two nodes and an edge that is not in the graph.

    ``_load_graph`` read every row as an edge, so the ``source,target`` first line
    that ``quvine --help`` and the QuVINE docs both show was loaded as an edge
    between a node named "source" and a node named "target". On a 34-node karate
    graph the run reported 34 nodes when the header was absent and 36 when it was
    present, embedded the two phantom nodes, and exited 0 either way. Nothing in
    the output said which had happened.

    These import pandas and networkx, which are base dependencies, not part of the
    ``[quvine]`` extra -- ``_load_graph`` deliberately imports them inside the
    function so ``--help`` still works on a bare install.
    """

    @staticmethod
    def _write(tmp_path, text):
        path = tmp_path / "edges.csv"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _load(self, path, **kwargs):
        from qbiocode.apps.quvine.cli import _load_graph

        return _load_graph(path, ",", False, **kwargs)

    def test_a_conventional_header_is_dropped(self, tmp_path):
        graph = self._load(self._write(tmp_path, "source,target\n0,1\n1,2\n"))
        assert sorted(graph.nodes) == ["0", "1", "2"]

    def test_a_file_without_a_header_is_unchanged(self, tmp_path):
        graph = self._load(self._write(tmp_path, "0,1\n1,2\n"))
        assert sorted(graph.nodes) == ["0", "1", "2"]

    def test_other_conventional_spellings_are_recognized(self, tmp_path):
        for first, second in [("from", "to"), ("gene1", "gene2"), ("Node_A", "node_b")]:
            graph = self._load(self._write(tmp_path, f"{first},{second}\nx,y\n"))
            assert sorted(graph.nodes) == ["x", "y"], f"{first},{second} not detected"

    def test_real_node_names_are_never_mistaken_for_a_header(self, tmp_path):
        """The rule is a closed set of names, not a heuristic, for exactly this.

        A sniffing rule -- "row 0's endpoints appear nowhere else", or "row 0 is
        non-numeric" -- silently deletes the first edge of a sparse
        string-labelled graph. Deleting a real edge is worse than the bug being
        fixed, because there is no phantom node to notice.
        """
        graph = self._load(self._write(tmp_path, "TP53,MDM2\nEGFR,KRAS\n"))
        assert sorted(graph.nodes) == ["EGFR", "KRAS", "MDM2", "TP53"]
        assert graph.number_of_edges() == 2

    def test_header_yes_forces_the_drop(self, tmp_path):
        graph = self._load(self._write(tmp_path, "TP53,MDM2\nEGFR,KRAS\n"), header="yes")
        assert sorted(graph.nodes) == ["EGFR", "KRAS"]

    def test_header_no_forces_the_row_to_be_read_as_an_edge(self, tmp_path):
        graph = self._load(self._write(tmp_path, "source,target\n0,1\n"), header="no")
        assert sorted(graph.nodes) == ["0", "1", "source", "target"]

    def test_a_header_only_file_is_an_error_not_an_empty_graph(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="only a header row"):
            self._load(self._write(tmp_path, "source,target\n"))

    def test_the_cli_exposes_the_override(self):
        """``--header`` must exist, or the escape hatch is unreachable."""
        tree = ast.parse((QUVINE_ROOT / "cli.py").read_text(encoding="utf-8"))
        flags = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        assert "--header" in flags
