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

"""Secrets read from protected files must not come back out somewhere unprotected.

``get_creds`` reads an IBM Quantum API token from ``~/.qiskit/qiskit-ibm.json`` -- a file
the user deliberately keeps outside any repository -- and used to ``print`` the assembled
dictionary, token included, on every call. That takes a secret from a protected location
and puts it on stdout, from where it reaches a terminal scrollback, a CI log, a pasted
issue report, and -- because the tutorial notebooks in this tree are committed *with their
outputs* and published -- potentially a public web page.

No notebook in the tree had actually leaked one: the only notebook that mentions
``get_creds`` runs on the simulator, so it never called it. These tests are what keep that
true rather than lucky.

Two properties are pinned:

* :func:`qbiocode.utils.ibm_account.redacted` removes secret *values* while keeping the
  rest, because "which instance did it resolve?" is the useful half of the diagnostic and
  needs no secret to answer.
* ``get_creds`` actually routes its output through it. Asserted by capturing real stdout
  with a token present, not by reading the source -- a redaction helper that exists and is
  not called is the failure mode worth catching.

The sibling guard for TabPFN's token, and the check that no committed notebook output
carries a token-shaped value at all, live in ``tests/test_tabpfn_token.py``.
"""

import json

from qbiocode.utils.ibm_account import _SECRET_KEYS, get_creds, redacted

#: Distinctive, and sharing no prefix with any key name or path in the assertions below.
FAKE_TOKEN = "SECRETKEY-ibm-4d7b2f9a1c3e"

FULL_CREDS = {
    "channel": "ibm_quantum_platform",
    "instance": "ibm-q/open/main",
    "token": FAKE_TOKEN,
    "url": "https://api.quantum.ibm.com",
}


class TestRedacted:
    def test_the_token_value_is_replaced(self):
        result = redacted(FULL_CREDS)
        assert result["token"] == "<redacted>"
        assert FAKE_TOKEN not in json.dumps(result)

    def test_not_even_a_prefix_survives(self):
        """A partial key is still a disclosure, and truncation is the tempting shortcut."""
        rendered = json.dumps(redacted(FULL_CREDS))
        for length in (4, 6, 8, 12):
            assert FAKE_TOKEN[:length] not in rendered, (
                f"the first {length} characters of the token survived redaction"
            )

    def test_the_useful_diagnostic_is_kept(self):
        """Redaction must not reduce this to uselessness -- the point is still to debug."""
        result = redacted(FULL_CREDS)
        assert result["channel"] == "ibm_quantum_platform"
        assert result["instance"] == "ibm-q/open/main"
        assert result["url"] == "https://api.quantum.ibm.com"

    def test_the_presence_of_a_token_is_still_visible(self):
        """'Did it find my credentials?' is answerable without the value."""
        assert "token" in redacted(FULL_CREDS)
        assert redacted(FULL_CREDS)["token"] != FULL_CREDS.get("missing")

    def test_an_absent_token_is_not_marked_as_present(self):
        """Marking a missing token '<redacted>' would invert the diagnostic."""
        assert redacted({"channel": "x", "token": None})["token"] is None
        assert redacted({"channel": "x", "token": ""})["token"] == ""
        assert "token" not in redacted({"channel": "x"})

    def test_the_input_is_not_mutated(self):
        """Callers pass the live dict on to QiskitRuntimeService straight afterwards."""
        original = dict(FULL_CREDS)
        redacted(FULL_CREDS)
        assert FULL_CREDS == original, "redacted() mutated the credentials it was given"

    def test_an_empty_dict_is_handled(self):
        assert redacted({}) == {}

    def test_token_is_among_the_declared_secrets(self):
        """A future credential field should join _SECRET_KEYS rather than be forgotten."""
        assert "token" in _SECRET_KEYS


class TestGetCredsDoesNotPrintTheToken:
    """The property that actually matters: the helper is used, not merely present."""

    def test_a_token_from_the_config_is_not_printed(self, capsys):
        creds = get_creds({"ibm_token": FAKE_TOKEN, "ibm_channel": "ibm_quantum_platform"})
        captured = capsys.readouterr()
        assert creds["token"] == FAKE_TOKEN, "the caller must still receive the real token"
        assert FAKE_TOKEN not in captured.out, (
            "get_creds printed the API token to stdout, which is how it reaches a CI log, "
            "a pasted issue report, or a committed notebook output"
        )
        assert FAKE_TOKEN not in captured.err
        assert "<redacted>" in captured.out, "the redacted diagnostic should still appear"

    def test_a_token_from_a_qiskit_json_file_is_not_printed(self, tmp_path, capsys):
        """The path that matters most: the token came from a file kept out of the repo."""
        qiskit_json = tmp_path / "qiskit-ibm.json"
        qiskit_json.write_text(json.dumps({
            "default-account": {
                "channel": "ibm_quantum_platform",
                "instance": "ibm-q/open/main",
                "token": FAKE_TOKEN,
            }
        }))
        creds = get_creds({"qiskit_json_path": str(qiskit_json)})
        captured = capsys.readouterr()
        assert creds.get("token") == FAKE_TOKEN, "credentials must still be resolved"
        assert FAKE_TOKEN not in captured.out
        assert FAKE_TOKEN[:8] not in captured.out

    def test_the_resolved_instance_is_still_reported(self, tmp_path, capsys):
        """Whoever added the print wanted to see this; redaction keeps it."""
        qiskit_json = tmp_path / "qiskit-ibm.json"
        qiskit_json.write_text(json.dumps({
            "default-account": {"channel": "ibm_quantum_platform",
                                "instance": "ibm-q/open/main", "token": FAKE_TOKEN}
        }))
        get_creds({"qiskit_json_path": str(qiskit_json)})
        assert "ibm-q/open/main" in capsys.readouterr().out

    def test_a_missing_file_still_says_so_without_raising(self, tmp_path, capsys):
        creds = get_creds({"qiskit_json_path": str(tmp_path / "absent.json")})
        assert creds == {}
        assert "not found" in capsys.readouterr().out.lower()

    def test_no_source_file_prints_nothing_secret(self, capsys):
        get_creds({})
        out = capsys.readouterr().out
        assert FAKE_TOKEN not in out


class TestNothingElseEchoesCredentials:
    """A grep-level guard, so a reintroduced raw print is caught at review time."""

    def test_get_creds_does_not_print_its_dict_directly(self):
        import inspect

        source = inspect.getsource(get_creds)
        assert "print(rval)" not in source, (
            "get_creds prints the raw credentials dict again; route it through redacted()"
        )
        assert "redacted(rval)" in source

    def test_instantiate_runtime_service_adds_no_second_print(self):
        import inspect

        from qbiocode.utils.ibm_account import instantiate_runtime_service

        source = inspect.getsource(instantiate_runtime_service)
        assert "print(" not in source, (
            "instantiate_runtime_service prints something; it handles credentials, so "
            "anything it emits needs the same scrutiny as get_creds"
        )
