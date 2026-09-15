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

"""Reading the TabPFN API token from a file that cannot be committed.

QBioCode's pinned model version (``v2``) needs no token at all, so most users never
reach this module. It exists for the newer checkpoints -- ``v2.5``, ``v2.6``, ``v3`` --
whose weights sit behind a non-commercial license acceptance, and for which the only
non-interactive authorisation is the ``TABPFN_TOKEN`` environment variable. Exporting
that by hand works but does not survive a new shell, a notebook kernel started from a
launcher, or a scheduled run -- so the token ends up pasted into a config file, a
notebook cell, or a shell profile that is inside a repository. This module gives it one
place to live instead, and that place is deliberately **outside the repository**.

Before using it, note that opting into those versions is a licensing decision: see
:data:`qbiocode.learning.compute_tabpfn.TABPFN_DEFAULT_VERSION`.

Why outside rather than gitignored
----------------------------------
A gitignored file inside the checkout is *unlikely* to be committed, not unable to be:
``git add -f`` overrides it, a rewritten ``.gitignore`` stops covering it, ``git stash
-u`` picks it up, and a file copied into a sibling clone is not covered at all. None of
those can reach ``~/.config/qbiocode/``. The ignore rules added alongside this module are
belt-and-braces for a token someone puts in the tree anyway, not the primary defence.

What this module will not do
----------------------------
It never returns, logs or prints the token. :func:`describe_token_source` exists so a
notebook can show *whether* a token is configured and *where it came from* without putting
key material into a committed output -- these pages are published, and a partial key is
still a disclosure. That is also why no fingerprint helper is provided: it would be one
convenience away from appearing on a rendered page.

The token is handed to TabPFN the only way TabPFN accepts it, by setting
``os.environ['TABPFN_TOKEN']``. Nothing here re-implements or intercepts the download.

Config key
----------
``tabpfn_json_path`` in a QProfiler config, mirroring the existing ``qiskit_json_path``
that :mod:`qbiocode.utils.ibm_account` reads IBM credentials from.
"""

import json
import os
import pathlib
import stat
import warnings

#: Where the token is looked for when no path is configured. One location, in the
#: XDG-style config directory, so there is a single answer to "where does it live".
DEFAULT_TOKEN_PATH = pathlib.Path("~/.config/qbiocode/tabpfn.json")

#: Keys accepted inside the JSON object. ``token`` is the documented one; the others are
#: tolerated because they are what someone naturally writes after reading TabPFN's own
#: instructions, and failing on a spelling difference would be a pointless obstacle.
_TOKEN_KEYS = ("token", "TABPFN_TOKEN", "tabpfn_token", "api_key")

#: What :func:`write_token_template` puts in the file. Recognised on read and reported as
#: "not configured", so a template that was created but never filled in does not look
#: like a live token that TabPFN then rejects for no visible reason.
PLACEHOLDER = "paste-your-tabpfn-api-key-here"

#: The environment variable TabPFN itself reads.
ENV_VAR = "TABPFN_TOKEN"


def _resolve_path(args=None, path=None):
    """The token file to use: explicit argument, then config key, then the default."""
    if path is not None:
        return pathlib.Path(path).expanduser()
    if args:
        configured = args.get("tabpfn_json_path")
        if configured:
            return pathlib.Path(configured).expanduser()
    return DEFAULT_TOKEN_PATH.expanduser()


def _warn_if_readable_by_others(path):
    """A credentials file readable by other accounts is worth saying so about.

    Not an error: on a single-user laptop it is harmless, and refusing to proceed would
    be obstructive. But the permissions are the only thing protecting the token from
    every other process running as another user on a shared machine.
    """
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover - raced away between exists() and stat()
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        warnings.warn(
            f"{path} is readable by other users (mode {stat.filemode(mode)}). It holds an "
            f"API token; restrict it with:  chmod 600 {path}",
            UserWarning,
            stacklevel=3,
        )


def _read_token(path):
    """Pull the token out of the JSON file, or raise a message naming the file.

    A file that exists was meant to work, so a malformed one is reported rather than
    skipped: silently falling through leaves the user with TabPFN's license error and no
    reason to suspect the file they just wrote.
    """
    try:
        text = path.read_text()
    except OSError as error:
        raise OSError(f"Cannot read the TabPFN token file {path}: {error}") from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} is not valid JSON ({error}). It should contain a single object:\n"
            f'  {{"token": "<your api key>"}}'
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} should contain a JSON object, not {type(payload).__name__}. "
            f'Write it as:  {{"token": "<your api key>"}}'
        )

    for key in _TOKEN_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"{path} contains no token. Expected one of {list(_TOKEN_KEYS)} as a key, but the "
        f"file has {sorted(payload)}. Write it as:  {{\"token\": \"<your api key>\"}}"
    )


def load_tabpfn_token(args=None, path=None, *, override=False):
    """Put the TabPFN token into the environment so TabPFN can find it.

    Args:
        args (dict or None): A QProfiler config. ``args['tabpfn_json_path']`` is used
            when present, mirroring ``qiskit_json_path``.
        path (str or Path or None): An explicit file, taking precedence over ``args``.
        override (bool): Replace a ``TABPFN_TOKEN`` that is already exported. Off by
            default: an environment variable is a deliberate, more specific act than a
            file left on disk, and silently overriding it makes a run untraceable.

    Returns:
        str or None: A short description of where the token came from -- suitable for
        printing, and containing no key material -- or ``None`` when no token was found.
        Absence is a state, not an error: the caller decides whether it matters, and
        ``compute_tabpfn`` already explains itself if it does.

    Raises:
        ValueError: If the file exists but is malformed or holds no token.
        OSError: If the file exists but cannot be read.
    """
    if not override and os.environ.get(ENV_VAR, "").strip():
        return f"{ENV_VAR} already set in the environment"

    token_path = _resolve_path(args, path)
    if not token_path.is_file():
        return None

    _warn_if_readable_by_others(token_path)
    token = _read_token(token_path)
    if token == PLACEHOLDER:
        warnings.warn(
            f"{token_path} still contains the placeholder token, so TabPFN cannot "
            f"authenticate. Replace {PLACEHOLDER!r} with the API key from "
            f"https://ux.priorlabs.ai (Account page).",
            UserWarning,
            stacklevel=2,
        )
        return None

    os.environ[ENV_VAR] = token
    return f"loaded from {_display(token_path)}"


def describe_token_source(args=None, path=None):
    """Whether a token is configured, and from where -- with no key material.

    Built for a notebook cell whose output gets committed and published. It reports the
    *state*, never the value, and returns no fingerprint or prefix of the token either.

    Returns:
        dict: ``configured`` (bool), ``source`` (str or None), ``path`` (str, the file
        that would be read, shown relative to home), ``exists`` (bool).
    """
    token_path = _resolve_path(args, path)
    from_env = bool(os.environ.get(ENV_VAR, "").strip())
    exists = token_path.is_file()

    configured = from_env
    source = f"{ENV_VAR} environment variable" if from_env else None
    if not configured and exists:
        try:
            configured = _read_token(token_path) != PLACEHOLDER
        except (OSError, ValueError):
            configured = False
        source = _display(token_path) if configured else None

    return {
        "configured": configured,
        "source": source,
        "path": _display(token_path),
        "exists": exists,
    }


def write_token_template(path=None, args=None):
    """Create the token file with a placeholder, and lock its permissions down.

    Creates the parent directory, writes ``{"token": PLACEHOLDER}`` and chmods the file
    to ``0600`` so that only the owner can read it.

    Refuses to touch an existing file: it may already hold a real token, and overwriting
    a credential to "help" is not recoverable.

    Returns:
        pathlib.Path: The file written.

    Raises:
        FileExistsError: If the file is already there.
    """
    token_path = _resolve_path(args, path)
    if token_path.exists():
        raise FileExistsError(
            f"{token_path} already exists and was left untouched, in case it holds a real "
            f"token. Edit it directly, or delete it first if you mean to start over."
        )
    token_path.parent.mkdir(parents=True, exist_ok=True)
    # Written through an exclusive open at 0600 rather than write_text() then chmod, so
    # the token the user pastes in is never briefly world-readable.
    descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"token": PLACEHOLDER}, handle, indent=2)
        handle.write("\n")
    return token_path


def check_tabpfn_access(args=None, path=None, *, timeout=10):
    """Report which gate is actually blocking TabPFN's weights.

    There are three independent gates, and upstream's error message conflates them. When
    a valid token is present but the license has not been accepted, ``ensure_license_accepted``
    falls through to a browser login, fails for want of a TTY, and reports::

        TabPFN requires a one-time license acceptance to download model weights for local
        inference, but no interactive terminal is available.
        ...
        4. Set the environment variable: export TABPFN_TOKEN="<your-api-key>"

    -- advising a step that is already done. The token was found and verified; what is
    missing is the *acceptance*, which is a separate action on the account. This function
    says which of the three it is, so the next person does not have to read
    ``tabpfn.browser_auth`` to find out.

    Makes up to two network calls (Prior Labs' auth API, and HuggingFace for the license
    name). Never sends or returns the token itself.

    Args:
        args (dict or None): A QProfiler config, for ``tabpfn_json_path``.
        path (str or Path or None): An explicit token file.
        timeout (int): Per-request timeout in seconds.

    Returns:
        dict: ``token`` (one of ``'absent'``, ``'invalid'``, ``'valid'``, ``'unknown'``),
        ``license_accepted`` (True, False or None), ``blocker`` (``None`` when nothing is
        blocking, else ``'extra'``, ``'token'``, ``'license'`` or ``'network'``) and
        ``advice`` (a sentence naming the next action). No token material.
    """
    import json as _json
    import urllib.request

    from qbiocode.learning.compute_tabpfn import tabpfn_is_available

    result = {"token": "absent", "license_accepted": None, "blocker": None, "advice": ""}

    if not tabpfn_is_available():
        result["blocker"] = "extra"
        result["advice"] = 'Install the extra:  pip install "qbiocode[tabpfn]"'
        return result

    load_tabpfn_token(args, path)
    from tabpfn.browser_auth import check_license_accepted, get_cached_token, verify_token
    from tabpfn.settings import settings

    token = get_cached_token()
    if not token:
        result["blocker"] = "token"
        result["advice"] = (
            "No API key found. Put one in "
            f"{_display(_resolve_path(args, path))} as {{\"token\": \"...\"}} -- see "
            "write_token_template()."
        )
        return result

    api_url = settings.tabpfn.auth_api_url
    status = verify_token(token, api_url)
    if status is None:
        result["token"] = "unknown"
        result["blocker"] = "network"
        result["advice"] = f"Could not reach {api_url} to verify the key. Check connectivity."
        return result
    if status is False:
        result["token"] = "invalid"
        result["blocker"] = "token"
        result["advice"] = (
            "The API key was rejected as invalid or expired. Copy a fresh one from "
            "https://ux.priorlabs.ai/account into the token file."
        )
        return result

    result["token"] = "valid"

    # The license name is per model version and lives on the HuggingFace model card.
    try:
        url = "https://huggingface.co/api/models/Prior-Labs/tabpfn_3"
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            card = _json.loads(response.read())
        license_name = card.get("cardData", {}).get("license_name")
    except Exception:  # noqa: BLE001 -- a diagnostic must not raise
        license_name = None

    if not license_name:
        result["blocker"] = "network"
        result["advice"] = (
            "The key is valid, but the model card could not be read to find which license "
            "version applies. Check connectivity to huggingface.co."
        )
        return result

    accepted = check_license_accepted(token, api_url, license_name)
    result["license_accepted"] = accepted
    if accepted is True:
        result["advice"] = "Nothing is blocking TabPFN; the weights will download on first fit."
        return result
    if accepted is None:
        result["blocker"] = "network"
        result["advice"] = "Could not reach the license server to check acceptance."
        return result

    result["blocker"] = "license"
    result["advice"] = (
        f"The API key is valid, but the {license_name} license has not been accepted on "
        f"this account. An API key authenticates you; it does not accept the license for "
        f"you. Open https://ux.priorlabs.ai, go to the Licenses tab and accept it -- then "
        f"nothing further is needed here. This step cannot be automated: it is an "
        f"agreement tied to your account."
    )
    return result


def _display(path):
    """A path safe to print: relative to home where possible, so no user name leaks.

    The notebooks in this tree are published with their outputs, and
    ``tests/test_docs_structure.py`` fails a build that commits an absolute local path.
    """
    path = pathlib.Path(path)
    try:
        return "~/" + str(path.relative_to(pathlib.Path.home()))
    except ValueError:
        return path.name
