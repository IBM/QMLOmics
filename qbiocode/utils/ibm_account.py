# This will be a simple function to extract information from a user's qiskit-json file

import json
import os

from qiskit_ibm_runtime import QiskitRuntimeService

#: Credential keys whose value must never be printed, logged or written to a results
#: directory. ``get_creds`` reads the token out of ``~/.qiskit/qiskit-ibm.json`` -- a file
#: the user deliberately keeps outside the repository -- so echoing it to stdout moves a
#: secret from a protected location into an unprotected one. That mattered here in
#: particular because the tutorial notebooks are committed *with their outputs* and
#: published, so one notebook calling this function would put a live API token on a public
#: page. See qbiocode.utils.tabpfn_account for the same principle applied from the start.
_SECRET_KEYS = frozenset({"token"})


def redacted(creds):
    """A copy of a credentials dict that is safe to print.

    The secret values are replaced with a marker rather than dropped: whether a token was
    found is the useful half of the diagnostic -- "did it pick up my credentials?" -- and
    it can be answered without disclosing the value.

    Args:
        creds (dict): Credentials as assembled by :func:`get_creds`.

    Returns:
        dict: The same keys, with every secret value replaced by ``'<redacted>'``.
    """
    return {
        key: ("<redacted>" if key in _SECRET_KEYS and value else value)
        for key, value in creds.items()
    }


def get_creds(args):
    """This function determines the user's IBM Quantum channel, instance, and token, using values provided
    within the config.yaml file or as defined within the user's qiskit configuration from provided qiskit_json_path
    specified in the config.yaml file, and then parses its contents. It returns the main items in this json file,
    such as the instance and api token, which can then be passed into the QML functions when using a real
    hardware backend.
    The function will return a dictionary with the keys 'channel', 'instance', 'token', and 'url',
    which can be used to instantiate the QiskitRuntimeService.
    If the qiskit_json_path is provided, it will attempt to read the credentials from that file.
    Args:
        args (dict): This passes the arguments from the config.yaml file.  In this particular case, it is importing the path to the qiskit-ibm.json file (qiskit_json_path) and the credentials
        defined in this json file (ibm_channel, ibm_instance, ibm_token, ibm_url).

    Returns:
        rval (dict): A dictionary containing the IBM Quantum credentials, including 'channel', 'instance', 'token', and 'url'.
    """
    cred_source_dict = {
        "channel": "ibm_channel",
        "instance": "ibm_instance",
        "token": "ibm_token",
        "url": "ibm_url",
        "name": "name"
    }
    rval = {}
    account_name = args.get("name", None)
    for ibm_name, yaml_name in cred_source_dict.items():
        value = args.get(yaml_name, None)
        if value:
            rval[ibm_name] = value

    qiskit_json_path = args.get("qiskit_json_path", None)
    if qiskit_json_path:
        qiskit_json_path = os.path.expanduser(qiskit_json_path)
        if os.path.exists(qiskit_json_path):
            # load the qiskit json file
            with open(qiskit_json_path, "r") as jfile:
                creds = json.load(jfile)
            # Access keys and values
            # The items we want are actually in a nested dictionary, so we have to loop through the outer dictionary first, then the
            # nested one.  This nested dictionary (outer_value) is actually the value for the key in the parent dictionary.
            selected_creds = None
            if account_name:
                selected_creds = creds.get(account_name, None)
            elif len(creds) == 1:
                selected_creds = next(iter(creds.values()))

            if selected_creds:
                for ibm_name in cred_source_dict.keys():
                    if ibm_name not in rval:
                        value = selected_creds.get(ibm_name, None)
                        if value:
                            rval[ibm_name] = value
        else:
            print(
                "IBM credentials not found! Please verify that the path to your qiskit-ibm.json file is correct."
            )
    # Redacted, not raw: `rval` carries the API token, and this print used to put it on
    # stdout on every call. See `redacted` above for why that is worse than it looks.
    print(redacted(rval))
    return rval


def instantiate_runtime_service(args):
    """This function provides a quick way to instantiate QiskitRuntimeService in one place. A basic call to this function can then be done in anywhere else.
    It uses the get_creds function to retrieve the necessary credentials from the qiskit-ibm.json file, with the file path specified in the config.yaml file.
    It returns an instance of the QiskitRuntimeService class, which can be used to interact with IBM Quantum services.

    Args:
        args (dict): This passes the arguments from the config.yaml file.  In this particular case, it is importing the path to the qiskit-ibm.json file (qiskit_json_path) and the credentials
        defined in this json file (ibm_channel, ibm_instance, ibm_token, ibm_url).

    Returns:
        QiskitRuntimeService: An instance of the QiskitRuntimeService class, initialized with the credentials from the qiskit-ibm.json file or the provided arguments.
    """
    return QiskitRuntimeService(**get_creds(args))
