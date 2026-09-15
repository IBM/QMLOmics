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

"""Guards on the packaging metadata.

The dependency tiers are declared twice by necessity: ``pyproject.toml`` needs
them as PEP 621 extras so that ``pip install "qbiocode[quvine]"`` works, and
``requirements/*.txt`` needs them so that ``pip install -r`` works. These tests
keep the two from drifting apart, and pin down the properties that were
previously wrong (see the "Packaging and dependency declarations" section of
CHANGELOG.md).
"""

import pathlib

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = REPO_ROOT / "requirements"


def read_pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def read_requirements(name):
    """Return the requirement specifiers in ``requirements/<name>``.

    Comments, blank lines, ``-r`` includes and ``-e`` installs are skipped, so
    the result is directly comparable to a PEP 621 extra's dependency list.
    """
    lines = (REQUIREMENTS_DIR / name).read_text().splitlines()
    specs = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        specs.append(line)
    return specs


@pytest.mark.parametrize(
    ("extra", "filename"),
    [
        ("quvine", "requirements-quvine.txt"),
        ("tabpfn", "requirements-tabpfn.txt"),
        ("docs", "requirements-docs.txt"),
    ],
)
def test_extra_matches_its_requirements_file(extra, filename):
    declared = read_pyproject()["project"]["optional-dependencies"][extra]
    assert sorted(declared) == sorted(read_requirements(filename)), (
        f"pyproject.toml's [{extra}] extra and requirements/{filename} have "
        "drifted apart; update whichever one is stale."
    )


def test_dynamic_dependencies_point_at_the_shipped_base_file():
    pyproject = read_pyproject()
    assert "dependencies" in pyproject["project"]["dynamic"]
    files = pyproject["tool"]["setuptools"]["dynamic"]["dependencies"]["file"]
    assert files == ["requirements/requirements-base.txt"]
    # A build reads this file, so it must exist and be non-empty or the built
    # package declares no dependencies at all.
    assert read_requirements("requirements-base.txt")


def test_all_extra_is_a_union_not_a_copy():
    """``all`` must reference the other extras rather than duplicate them.

    The previous hand-maintained copy had already drifted: it carried `pandoc`
    and omitted nothing it should have, but nothing kept it honest.
    """
    extras = read_pyproject()["project"]["optional-dependencies"]
    assert extras["all"] == ["qbiocode[apps,quvine,tabpfn,docs,dev]"]
    for name in ("apps", "quvine", "tabpfn", "dev", "docs"):
        assert name in extras


def test_build_tooling_is_not_a_runtime_dependency():
    """A build-time pin declared as a runtime dependency is a metadata bug.

    ``setuptools<81`` is required by ``node2vec`` and so legitimately belongs to
    the ``quvine`` extra, but it must never reach the unconditional list.
    """
    base = read_requirements("requirements-base.txt")
    assert not [spec for spec in base if spec.lower().startswith(("setuptools", "wheel"))]
    assert "setuptools<81" in read_pyproject()["project"]["optional-dependencies"]["quvine"]


def test_library_imports_are_declared():
    """Every third-party module imported under ``qbiocode/`` must be declared.

    ``pyyaml``, ``matplotlib`` and ``joblib`` were previously imported but
    undeclared, and only resolved because other dependencies pulled them in.
    """
    base = {spec.split("=")[0].split("<")[0].split(">")[0].strip().lower() for spec in
            read_requirements("requirements-base.txt")}
    # Distribution name differs from the import name for these.
    for distribution in ("pyyaml", "matplotlib", "joblib", "scikit-learn", "umap-learn"):
        assert distribution in base, f"{distribution} is imported by qbiocode/ but not declared"


def test_pandoc_is_not_a_pip_dependency():
    """Pandoc is a system binary; the PyPI package of that name is not it."""
    extras = read_pyproject()["project"]["optional-dependencies"]
    assert not [spec for spec in extras["docs"] if spec.lower().startswith("pandoc")]
    assert not [spec for spec in read_requirements("requirements-docs.txt")
                if spec.lower().startswith("pandoc")]


def test_tensorflow_is_not_declared():
    """Nothing in the tree imports TensorFlow or Keras."""
    base = read_requirements("requirements-base.txt")
    assert not [spec for spec in base if spec.lower().startswith(("tensorflow", "keras"))]


def test_root_requirements_txt_delegates_to_the_tiered_files():
    """``pip install -r requirements.txt`` is documented in 4 places; keep it working."""
    text = (REPO_ROOT / "requirements.txt").read_text()
    assert "-r requirements/requirements.txt" in text


def test_setup_py_declares_no_metadata():
    """pyproject.toml is the single source of metadata; setup.py is a shim."""
    text = (REPO_ROOT / "setup.py").read_text()
    assert "setup()" in text
    for field in ("install_requires", "extras_require", "entry_points", "packages="):
        assert field not in text, f"setup.py re-declares {field}, which pyproject.toml owns"


def test_no_lock_file_is_committed():
    """A lock file is one OS/arch/Python; committing one implies it covers all nine.

    The inherited lock also outlived its own dependency set -- it still pinned
    tensorflow long after the tree stopped importing it. The recipe for generating
    one lives in ``requirements/requirements.txt``; the artifact stays local.
    """
    lock = REQUIREMENTS_DIR / "requirements-lock.txt"
    assert not lock.exists() or _is_git_ignored(lock), (
        "requirements/requirements-lock.txt is tracked; a pip freeze is specific to "
        "one platform and Python version and must not be committed"
    )
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "requirements/requirements-lock.txt" in gitignore
    assert "pip freeze" in (REQUIREMENTS_DIR / "requirements.txt").read_text(), (
        "the lock-generation recipe is gone from requirements/requirements.txt"
    )


def _is_git_ignored(path):
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT, capture_output=True
    )
    return result.returncode == 0


class TestTheCiProvesTheExtraIsOptional:
    """The `[quvine]` extra is only optional if something installs *without* it.

    The `test` job installs `.[dev]`, which brings the QuVINE dependencies along, so
    it cannot detect an eager `import gensim` on the import path. The `install-matrix`
    job's bare leg is what does; these tests keep it from being quietly dropped.
    """

    @staticmethod
    def workflow():
        import yaml
        with open(REPO_ROOT / ".github" / "workflows" / "ci.yml") as handle:
            return yaml.safe_load(handle)

    def test_a_job_installs_the_project_with_no_extras(self):
        legs = [
            leg["install"]
            for job in self.workflow()["jobs"].values()
            for leg in job.get("strategy", {}).get("matrix", {}).get("include", [])
            if "install" in leg
        ]
        assert "." in legs, (
            "no CI leg installs the project bare, so nothing checks that the "
            f"[quvine] extra is optional; legs found: {legs}"
        )
        assert ".[quvine]" in legs, f"no CI leg installs the extra; legs found: {legs}"

    def test_the_bare_leg_does_not_install_the_dev_extra(self):
        """`[dev]` would pull the very dependencies the leg checks are absent."""
        job = self.workflow()["jobs"]["install-matrix"]
        install_steps = "\n".join(step.get("run", "") for step in job["steps"])
        assert "[dev]" not in install_steps
        assert "pip install pytest" in install_steps, (
            "the bare leg needs pytest installed on its own, since [dev] is excluded"
        )

    def test_it_runs_the_console_scripts(self):
        """The regression guard for the broken `cli.py` import, at install level."""
        job = self.workflow()["jobs"]["install-matrix"]
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        for script in ("qprofiler --help", "qsage --help", "quvine --help"):
            assert script in runs, f"{script} is not smoke-tested by install-matrix"
