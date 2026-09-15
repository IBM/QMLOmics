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

"""Handling the TabPFN API token without leaking it.

A credentials helper has two jobs and the second one is easy to get wrong: read the
secret, and never let it out anywhere else. These tests cover both, and lean on the
second -- most of the failure modes here are disclosures rather than crashes, so they
would not announce themselves.

The specific things being pinned:

* The token file lives **outside the repository**, which is what makes committing it
  impossible rather than merely discouraged. The ``.gitignore`` rules are belt-and-braces
  for a file someone puts in the tree anyway, and are checked against real ``git
  check-ignore`` rather than by pattern-matching the file.
* Nothing that is safe-to-print ever contains token material -- not the value, not a
  prefix, not a fingerprint. The notebooks in this tree are published *with their
  committed outputs*, so a status line that included four characters of the key would put
  four characters of the key on a public page.
* An exported ``TABPFN_TOKEN`` wins over the file, because an environment variable is the
  more deliberate act, and silently overriding it makes a run impossible to explain.

Every test here manipulates ``TABPFN_TOKEN`` and the token path through fixtures, so none
of them reads or writes the developer's real ``~/.config/qbiocode/tabpfn.json``.
"""

import json
import os
import pathlib
import stat
import subprocess

import pytest

from qbiocode.utils.tabpfn_account import (
    DEFAULT_TOKEN_PATH,
    ENV_VAR,
    PLACEHOLDER,
    describe_token_source,
    load_tabpfn_token,
    write_token_template,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Stand-in for a real key. Deliberately shares no prefix with "tabpfn.json" or any other
#: path component: the disclosure tests below check that even the first few characters of
#: the token are absent from a status line, and a token beginning "tabpfn" would match the
#: *filename* and fail for a reason that has nothing to do with the code.
FAKE_TOKEN = "SECRETKEY-9f3c1e7a2b8d4056"


@pytest.fixture(autouse=True)
def no_ambient_token(monkeypatch):
    """Never let the developer's own exported token influence a test, or vice versa."""
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.fixture
def token_file(tmp_path):
    """A token file in a temporary directory, holding ``FAKE_TOKEN``."""
    path = tmp_path / "tabpfn.json"
    path.write_text(json.dumps({"token": FAKE_TOKEN}))
    path.chmod(0o600)
    return path


class TestReadingTheToken:
    def test_a_token_file_reaches_the_environment(self, token_file):
        """The environment variable is the only channel TabPFN itself reads."""
        source = load_tabpfn_token(path=token_file)
        assert source is not None
        assert os.environ[ENV_VAR] == FAKE_TOKEN

    def test_the_config_key_is_honoured(self, token_file):
        """``tabpfn_json_path`` mirrors the existing ``qiskit_json_path``."""
        load_tabpfn_token({"tabpfn_json_path": str(token_file)})
        assert os.environ[ENV_VAR] == FAKE_TOKEN

    def test_an_explicit_path_beats_the_config_key(self, tmp_path, token_file):
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"token": "not-this-one"}))
        load_tabpfn_token({"tabpfn_json_path": str(other)}, path=token_file)
        assert os.environ[ENV_VAR] == FAKE_TOKEN

    @pytest.mark.parametrize("key", ["token", "TABPFN_TOKEN", "tabpfn_token", "api_key"])
    def test_the_naturally_written_key_names_are_accepted(self, tmp_path, key):
        """Failing on a spelling difference would be a pointless obstacle."""
        path = tmp_path / "tabpfn.json"
        path.write_text(json.dumps({key: FAKE_TOKEN}))
        path.chmod(0o600)
        load_tabpfn_token(path=path)
        assert os.environ[ENV_VAR] == FAKE_TOKEN

    def test_a_missing_file_is_a_state_not_an_error(self, tmp_path):
        """The caller decides whether absence matters; compute_tabpfn already explains it."""
        assert load_tabpfn_token(path=tmp_path / "absent.json") is None
        assert ENV_VAR not in os.environ

    def test_an_exported_token_is_not_overridden(self, token_file, monkeypatch):
        """An environment variable is the more deliberate act, so it wins."""
        monkeypatch.setenv(ENV_VAR, "exported-by-hand")
        source = load_tabpfn_token(path=token_file)
        assert os.environ[ENV_VAR] == "exported-by-hand"
        assert "environment" in source

    def test_override_is_available_but_opt_in(self, token_file, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "exported-by-hand")
        load_tabpfn_token(path=token_file, override=True)
        assert os.environ[ENV_VAR] == FAKE_TOKEN

    def test_the_unfilled_template_is_not_mistaken_for_a_token(self, tmp_path):
        """Otherwise TabPFN rejects the placeholder and the user blames the wrong thing."""
        path = tmp_path / "tabpfn.json"
        path.write_text(json.dumps({"token": PLACEHOLDER}))
        path.chmod(0o600)
        with pytest.warns(UserWarning, match="placeholder"):
            assert load_tabpfn_token(path=path) is None
        assert ENV_VAR not in os.environ


class TestMalformedFilesAreReportedNotSkipped:
    """A file that exists was meant to work, so falling through silently misleads."""

    def test_invalid_json_names_the_file_and_the_shape(self, tmp_path):
        path = tmp_path / "tabpfn.json"
        path.write_text("{not json")
        path.chmod(0o600)
        with pytest.raises(ValueError) as exc:
            load_tabpfn_token(path=path)
        assert str(path) in str(exc.value)
        assert '"token"' in str(exc.value), "must show the expected shape"

    def test_a_json_object_with_no_token_lists_what_it_found(self, tmp_path):
        path = tmp_path / "tabpfn.json"
        path.write_text(json.dumps({"apikey": FAKE_TOKEN}))
        path.chmod(0o600)
        with pytest.raises(ValueError) as exc:
            load_tabpfn_token(path=path)
        assert "apikey" in str(exc.value)

    def test_a_bare_string_is_rejected_with_the_object_form(self, tmp_path):
        path = tmp_path / "tabpfn.json"
        path.write_text(json.dumps(FAKE_TOKEN))
        path.chmod(0o600)
        with pytest.raises(ValueError, match="JSON object"):
            load_tabpfn_token(path=path)

    def test_a_malformed_file_does_not_leak_its_contents(self, tmp_path):
        """The error names the file; it must not quote what was in it.

        A file whose JSON is broken *by a stray character in the key* would otherwise put
        the key into an exception message, and from there into a log or an issue report.
        """
        path = tmp_path / "tabpfn.json"
        path.write_text('{"token": "' + FAKE_TOKEN + '" oops}')
        path.chmod(0o600)
        with pytest.raises(ValueError) as exc:
            load_tabpfn_token(path=path)
        assert FAKE_TOKEN not in str(exc.value)


class TestNothingPrintableContainsTheToken:
    """The disclosure half, which is the half that fails silently."""

    def test_the_status_dict_holds_no_token_material(self, token_file):
        load_tabpfn_token(path=token_file)
        status = describe_token_source(path=token_file)
        assert status["configured"] is True
        rendered = json.dumps(status)
        assert FAKE_TOKEN not in rendered
        # Not even a prefix. A published page showing the first characters of a key is
        # still a disclosure, and it is the obvious "helpful" thing to add later.
        for length in (4, 6, 8):
            assert FAKE_TOKEN[:length] not in rendered, (
                f"the first {length} characters of the token appear in the status"
            )

    def test_the_source_description_holds_no_token_material(self, token_file):
        source = load_tabpfn_token(path=token_file)
        assert FAKE_TOKEN not in source
        assert FAKE_TOKEN[:4] not in source

    def test_the_status_path_is_relative_to_home(self, monkeypatch, tmp_path):
        """An absolute path names the user; a docs test already fails a build over it."""
        monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
        path = tmp_path / ".config" / "qbiocode" / "tabpfn.json"
        status = describe_token_source(path=path)
        assert status["path"].startswith("~/"), status["path"]
        assert str(tmp_path) not in status["path"]

    def test_status_works_and_stays_quiet_when_nothing_is_configured(self, tmp_path):
        status = describe_token_source(path=tmp_path / "absent.json")
        assert status == {
            "configured": False,
            "source": None,
            "path": (tmp_path / "absent.json").name,
            "exists": False,
        }

    def test_a_malformed_file_reports_unconfigured_rather_than_raising(self, tmp_path):
        """``describe_token_source`` is for a status line; it must never be the thing that
        breaks a notebook cell."""
        path = tmp_path / "tabpfn.json"
        path.write_text("{broken")
        path.chmod(0o600)
        assert describe_token_source(path=path)["configured"] is False


class TestTheTemplate:
    def test_it_is_created_owner_readable_only(self, tmp_path):
        path = write_token_template(path=tmp_path / "sub" / "tabpfn.json")
        assert path.is_file()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"created with {oct(mode)}, expected 0o600"
        assert json.loads(path.read_text()) == {"token": PLACEHOLDER}

    def test_it_refuses_to_overwrite_an_existing_file(self, token_file):
        """It might hold a real token, and overwriting a credential is not recoverable."""
        with pytest.raises(FileExistsError):
            write_token_template(path=token_file)
        assert json.loads(token_file.read_text())["token"] == FAKE_TOKEN

    def test_the_placeholder_is_not_a_plausible_key(self):
        """It must be obviously fake, so nobody ships it thinking it works."""
        assert "paste" in PLACEHOLDER and "here" in PLACEHOLDER

    def test_a_group_readable_file_is_flagged(self, tmp_path):
        path = tmp_path / "tabpfn.json"
        path.write_text(json.dumps({"token": FAKE_TOKEN}))
        path.chmod(0o644)
        with pytest.warns(UserWarning, match="readable by other users"):
            load_tabpfn_token(path=path)


class TestTheTokenCannotBeCommitted:
    """The primary defence is the location; the ignore rules are the backup."""

    def test_the_default_location_is_outside_the_repository(self):
        resolved = DEFAULT_TOKEN_PATH.expanduser()
        assert not str(resolved).startswith(str(REPO_ROOT)), (
            f"the default token path {resolved} is inside the checkout, where a "
            f"gitignore rule is the only thing standing between it and a commit"
        )
        assert resolved.name.endswith(".json")

    @pytest.mark.parametrize(
        "candidate",
        [
            "tabpfn.json",
            "tabpfn_config.json",
            "tabpfn_token.json",
            "tutorial/tabpfn.json",
            "qbiocode/tabpfn_config.json",
            "tests/tabpfn_token.json",
        ],
    )
    def test_a_token_placed_in_the_tree_anyway_is_ignored(self, candidate):
        """Checked with real ``git check-ignore``, not by reading the patterns.

        A pattern that looks right and does not match -- ``tabpfn.json`` without ``**/``
        covering subdirectories, say -- is exactly the failure this needs to catch.
        """
        result = subprocess.run(
            ["git", "check-ignore", "-q", candidate], cwd=REPO_ROOT, capture_output=True
        )
        assert result.returncode == 0, (
            f"{candidate} is not gitignored, so a token written there could be committed"
        )

    def test_no_token_file_is_tracked_right_now(self):
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.splitlines()
        offenders = [
            name
            for name in tracked
            if pathlib.Path(name).name.startswith("tabpfn")
            and name.endswith(".json")
        ]
        assert not offenders, f"a TabPFN token file is tracked by git: {offenders}"


class TestNoCommittedNotebookOutputCarriesASecret:
    """The notebooks are published with their outputs, so an output is a public page.

    Not specific to TabPFN: this is the general guard. ``ibm_account.get_creds`` prints the
    dict it assembled, IBM API token included, so a notebook that calls it and is then
    committed would publish that token. This test is what would catch it.
    """

    @staticmethod
    def notebooks():
        return sorted((REPO_ROOT / "tutorial").rglob("*.ipynb"))

    def test_at_least_one_notebook_is_being_checked(self):
        assert self.notebooks(), "no notebooks found; this guard would pass vacuously"

    def test_no_notebook_mentions_a_token_environment_variable_value(self):
        """An `os.environ['TABPFN_TOKEN']` echoed into a cell output would land here."""
        offenders = []
        for path in self.notebooks():
            text = path.read_text(encoding="utf-8")
            # The variable *name* is fine and expected -- the notebook documents it. A
            # long opaque string next to it is not.
            for marker in ("TABPFN_TOKEN=", "TABPFN_TOKEN':", 'TABPFN_TOKEN":'):
                index = text.find(marker)
                while index != -1:
                    tail = text[index + len(marker) : index + len(marker) + 48]
                    if any(ch.isalnum() for ch in tail) and "<" not in tail[:3]:
                        stripped = tail.strip(" \"'")
                        if len(stripped) > 16 and " " not in stripped[:16]:
                            offenders.append((path.name, marker, stripped[:24]))
                    index = text.find(marker, index + 1)
        assert not offenders, f"possible token values in notebooks: {offenders}"

    def test_no_notebook_carries_the_placeholder_alongside_a_real_looking_key(self):
        """The template value is safe; anything else in a `token` field is not."""
        import re

        pattern = re.compile(r'"(?:token|api_key|apikey)"\s*:\s*"([^"]{8,})"')
        offenders = []
        for path in self.notebooks():
            for value in pattern.findall(path.read_text(encoding="utf-8")):
                if value not in (PLACEHOLDER, "<your api key>", "..."):
                    offenders.append((path.name, value[:24]))
        assert not offenders, f"token-shaped values committed in notebooks: {offenders}"


class TestTheAccessDiagnostic:
    """Which of the three gates is blocking, told apart.

    This exists because upstream conflates them. With a valid token whose license has not
    been accepted, ``ensure_license_accepted`` falls through to a browser login, fails for
    want of a TTY, and advises setting ``TABPFN_TOKEN`` -- which is already set. The
    diagnostic is what turns that into "accept the license on your account".

    Every network call is mocked: a test suite must not depend on Prior Labs' API being
    reachable, and must never send a real token anywhere.
    """

    @staticmethod
    def _patch(monkeypatch, *, verify, accepted, license_name="tabpfn-3-license-v1.0"):
        pytest.importorskip("tabpfn")
        import json as _json
        import urllib.request

        import tabpfn.browser_auth as auth

        monkeypatch.setattr(auth, "get_cached_token", lambda: FAKE_TOKEN)
        monkeypatch.setattr(auth, "verify_token", lambda *a, **k: verify)
        monkeypatch.setattr(auth, "check_license_accepted", lambda *a, **k: accepted)

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return _json.dumps({"cardData": {"license_name": license_name}}).encode()

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())

    def test_a_missing_extra_is_named_as_the_blocker(self, monkeypatch):
        import importlib

        from qbiocode.utils import check_tabpfn_access

        # Resolved with import_module and patched on the module object rather than via a
        # dotted string target: `qbiocode/learning/__init__.py` re-exports the function
        # `compute_tabpfn`, which shadows the submodule of the same name, so
        # monkeypatch.setattr("qbiocode.learning.compute_tabpfn.tabpfn_is_available", ...)
        # resolves to the function and raises AttributeError.
        module = importlib.import_module("qbiocode.learning.compute_tabpfn")
        monkeypatch.setattr(module, "tabpfn_is_available", lambda: False)
        result = check_tabpfn_access()
        assert result["blocker"] == "extra"
        assert "qbiocode[tabpfn]" in result["advice"]

    def test_an_absent_token_is_named_as_the_blocker(self, monkeypatch, tmp_path):
        pytest.importorskip("tabpfn")
        import tabpfn.browser_auth as auth

        from qbiocode.utils import check_tabpfn_access

        monkeypatch.setattr(auth, "get_cached_token", lambda: None)
        result = check_tabpfn_access(path=tmp_path / "absent.json")
        assert result["blocker"] == "token"
        assert result["token"] == "absent"

    def test_a_rejected_token_is_distinguished_from_an_absent_one(self, monkeypatch):
        from qbiocode.utils import check_tabpfn_access

        self._patch(monkeypatch, verify=False, accepted=None)
        result = check_tabpfn_access()
        assert result["token"] == "invalid"
        assert result["blocker"] == "token"
        assert "rejected" in result["advice"]

    def test_the_real_case_a_valid_token_with_no_license_acceptance(self, monkeypatch):
        """The state this whole function exists for."""
        from qbiocode.utils import check_tabpfn_access

        self._patch(monkeypatch, verify=True, accepted=False)
        result = check_tabpfn_access()
        assert result["token"] == "valid"
        assert result["license_accepted"] is False
        assert result["blocker"] == "license"
        advice = result["advice"]
        assert "Licenses tab" in advice, "must say where to click"
        assert "does not accept the license for you" in advice, (
            "must correct the assumption that an API key is sufficient"
        )
        assert "cannot be automated" in advice, (
            "must be explicit that this is not something QBioCode can do for the user"
        )

    def test_nothing_blocking_is_reported_as_nothing_blocking(self, monkeypatch):
        from qbiocode.utils import check_tabpfn_access

        self._patch(monkeypatch, verify=True, accepted=True)
        result = check_tabpfn_access()
        assert result["blocker"] is None
        assert result["license_accepted"] is True

    @pytest.mark.parametrize(
        ("verify", "accepted"), [(None, None), (True, None)]
    )
    def test_an_unreachable_server_is_not_reported_as_a_bad_token(
        self, monkeypatch, verify, accepted
    ):
        """Blaming the key for a network failure sends the user to regenerate it."""
        from qbiocode.utils import check_tabpfn_access

        self._patch(monkeypatch, verify=verify, accepted=accepted)
        result = check_tabpfn_access()
        assert result["blocker"] == "network"
        assert result["token"] != "invalid"

    def test_the_diagnostic_never_returns_token_material(self, monkeypatch):
        from qbiocode.utils import check_tabpfn_access

        for verify, accepted in ((True, False), (True, True), (False, None), (None, None)):
            self._patch(monkeypatch, verify=verify, accepted=accepted)
            rendered = json.dumps(check_tabpfn_access())
            assert FAKE_TOKEN not in rendered
            assert FAKE_TOKEN[:6] not in rendered

    def test_it_does_not_raise_when_the_model_card_is_unreachable(self, monkeypatch):
        """A diagnostic that crashes is worse than useless -- it is another mystery."""
        pytest.importorskip("tabpfn")
        import urllib.request

        import tabpfn.browser_auth as auth

        from qbiocode.utils import check_tabpfn_access

        monkeypatch.setattr(auth, "get_cached_token", lambda: FAKE_TOKEN)
        monkeypatch.setattr(auth, "verify_token", lambda *a, **k: True)

        def _boom(*args, **kwargs):
            raise OSError("no network")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        result = check_tabpfn_access()
        assert result["blocker"] == "network"
        assert result["token"] == "valid"
