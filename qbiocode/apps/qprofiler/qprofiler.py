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

# ====== Base class imports ======
import numpy as np
import pandas as pd
import logging
from collections.abc import Sequence
import pickle
import os
import re
import csv
import time
# ====== Hydra imports ======
import hydra

# ====== Scikit-learn imports ======
from sklearn.model_selection import train_test_split

# ====== Qiskit imports ======
from qiskit_algorithms.utils import algorithm_globals

import sys

#: Best-effort guess at the checkout root, kept because ``folder_path`` in every
#: shipped config is written relative to it. The substitution only lands when the
#: current directory really does sit under one literally named ``QBioCode``, which
#: is not true of a GitHub source zip (``QBioCode-main``), a lowercase clone, or a
#: pip install -- hence ``_resolve_input_folder`` below rather than this alone.
#:
#: :meta private:
#:
#: Private to autodoc deliberately: its value is whatever directory the *documenting*
#: process ran in, so publishing it wrote the doc builder's own absolute filesystem
#: path into the API page on GitHub Pages.
dir_home = re.sub( 'QBioCode.*', 'QBioCode', os.getcwd() )
if os.path.isdir(dir_home):
    sys.path.append( dir_home )


def _resolve_input_folder(folder_path):
    """Return an existing directory for ``folder_path``, or ``None``.

    Candidates, in order:

    1. ``folder_path`` itself -- absolute, or relative to the current directory.
    2. ``folder_path`` under each ancestor of the current directory. This is what
       makes the tutorials work from a checkout whose top directory is not called
       ``QBioCode``: run from ``QBioCode-main/tutorial/QProfiler`` with
       ``folder_path: tutorial/QProfiler/data/ld_data``, candidate 1 points two
       levels too deep and the ``QBioCode``-derived root below does not exist,
       while the ancestor walk finds the real one.
    3. ``folder_path`` under a checkout root derived from the *current* directory.
    4. ``folder_path`` under ``dir_home``, the same root derived from the directory
       this module was imported from. Last, deliberately: ``dir_home`` is frozen at
       import time, so in a long-lived process (a notebook kernel, a test session)
       it can name a different checkout than the one the caller is standing in, and
       ranking it above the ancestor walk let a stale root silently win.
    """
    here = os.path.abspath(os.getcwd())
    candidates = [folder_path]
    while True:
        candidates.append(os.path.join(here, folder_path))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    derived = re.sub("QBioCode.*", "QBioCode", os.path.abspath(os.getcwd()))
    candidates.append(os.path.join(derived, folder_path))
    candidates.append(os.path.join(dir_home, folder_path))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None

# ====== Scaling and encoding functions imports ======
from qbiocode import scale_train_test, feature_encoding
from qbiocode import get_embeddings
from qbiocode.embeddings import check_embedding_name
# ====== Evaluation functions imports ====
#from qmlbench.evaluation.dataset_evaluation_no_var_threshold import evaluate2 # use this for moons/circles data, otherwise you'll run into an error with finding no features with minimum variance threshold
from qbiocode import evaluate
from qbiocode import model_run

#: Config keys ``main`` reads unconditionally. Reported together rather than one
#: KeyError at a time, so a hand-written config can be fixed in a single pass.
_REQUIRED_CONFIG_KEYS = (
    "folder_path", "file_dataset", "embeddings", "n_components", "model",
    "seed", "q_seed", "test_size", "iter", "scaling", "backend", "n_jobs",
)


def _resolve_scaling(scaling):
    """Resolve the ``scaling`` config value to a scaler name for ``scale_train_test``.

    The shipped config writes ``scaling: ['True']`` and the original code tested
    it with ``'True' in args['scaling']``. That substring test accepted
    ``'MinMaxScalerTrue'``, silently ignored ``['true']``, and raised
    ``TypeError: argument of type 'bool' is not iterable`` for the most natural
    YAML of all -- ``scaling: true``. All four spellings are accepted here, and
    anything else is named as an error instead of quietly disabling scaling.

    The single-element unwrapping tests ``Sequence`` rather than ``list``: Hydra
    hands the config over as ``omegaconf.ListConfig``, which is a ``Sequence``
    but *not* a ``list`` subclass, so an ``isinstance(value, list)`` test passes
    every dict-based unit test and then rejects the shipped config's own
    ``scaling: ['True']`` on the real CLI path.

    Returns:
        str: ``'MinMaxScaler'``, ``'StandardScaler'`` or ``'None'``.

    Raises:
        ValueError: if the value is not a recognized flag or scaler name.
    """
    value = scaling
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(
                f"scaling accepts a single value; got {list(value)!r}. Use "
                f"scaling: ['True'], scaling: false, or a scaler name."
            )
        value = value[0]
    if isinstance(value, bool):
        return "MinMaxScaler" if value else "None"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return "MinMaxScaler"
        if lowered in ("false", "no", "0", "none"):
            return "None"
        if lowered == "minmaxscaler":
            return "MinMaxScaler"
        if lowered == "standardscaler":
            return "StandardScaler"
    raise ValueError(
        f"Unrecognized scaling {scaling!r}. Accepted: true/false (or ['True'] / "
        f"['False'] as the shipped config writes it), 'MinMaxScaler', "
        f"'StandardScaler', or 'None'."
    )


def _validate_config(args, log):
    """Check the whole config before any dataset is read.

    A QProfiler run is long: loading data, splitting, embedding and fitting
    quantum models takes minutes to hours. Every check below used to fire deep
    into that run, or not at all -- an empty ``embeddings`` list, or an ``iter``
    of 0, simply produced no results and exited 0, which is indistinguishable
    from a run whose models all failed.

    Returns:
        str: the resolved scaler name, so ``main`` does not re-derive it.

    Raises:
        ValueError: naming the offending key, the value received, and what is
            accepted.
    """
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in args]
    if missing:
        raise ValueError(
            f"Config is missing required key(s): {missing}. Start from the "
            f"packaged configs/config.yaml, which defines all of "
            f"{list(_REQUIRED_CONFIG_KEYS)}."
        )

    if not isinstance(args["iter"], int) or isinstance(args["iter"], bool) or args["iter"] < 1:
        raise ValueError(
            f"iter is the number of train/test splits and must be a positive "
            f"integer; got {args['iter']!r}. A value of 0 produces no results at "
            f"all while still exiting successfully."
        )
    if not 0.0 < float(args["test_size"]) < 1.0:
        raise ValueError(
            f"test_size is a proportion and must be strictly between 0 and 1; "
            f"got {args['test_size']!r}."
        )
    n_components = args["n_components"]
    if (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise ValueError(
            f"n_components is the embedding width and must be a positive integer; "
            f"got {n_components!r}."
        )
    if not isinstance(args["n_jobs"], int) or args["n_jobs"] == 0:
        raise ValueError(
            f"n_jobs must be a non-zero integer (-1 means all cores); got "
            f"{args['n_jobs']!r}."
        )

    embeddings = list(args["embeddings"])
    if not embeddings:
        raise ValueError(
            "embeddings is empty; there is nothing to profile. Use ['none'] to "
            "run the models on the unreduced features."
        )
    # Validated as a set, up front: a typo in the last of six embeddings used to
    # surface only after the first five had been embedded and modelled.
    for name in embeddings:
        check_embedding_name(name)

    models = list(args["model"])
    if not models:
        raise ValueError(
            "model is empty; there is nothing to fit. See "
            "qbiocode.evaluation.model_run for the available model names."
        )

    scaler_name = _resolve_scaling(args["scaling"])
    log.info(f"Feature scaling resolved to: {scaler_name}")
    return scaler_name


# Begin the main function and instatiate Hydra class
# config_path=None allows --config-dir to work properly
def _append_model_row(path, row):
    """Append one model's result row to ``path``, widening the file if it has to.

    ``ModelResults.csv`` is written incrementally -- one row per model, as each model
    finishes -- so that a run interrupted half way still leaves usable output. The
    obvious way to do that, ``csv.writer`` plus a header written when the file is
    empty, is wrong as soon as two models contribute different keys, and QProfiler has
    two that routinely do: a *tuned* classical model reports ``BestParams_Tuned`` while
    an *untuned* one reports ``Model_Parameters``, and ``grid_search: True`` with
    ``tune_quantum: False`` -- the natural way to run a sweep, since a quantum fit per
    trial is expensive -- puts both in the same run.

    The result was a file whose header was narrower than its later rows, which
    ``pandas.read_csv`` refuses outright::

        ParserError: Expected 150 fields in line 7, saw 151

    so nothing downstream could read the results at all. Writing through
    :class:`csv.DictWriter` against the header already on disk fixes that: a row that
    introduces new columns rewrites the file with the union header and pads the earlier
    rows, and a row missing a column writes an empty cell there rather than shifting
    every later value left by one.

    Widening rewrites the file, which is ``O(rows)``, but it happens at most once per
    distinct key set -- in practice once, when the first untuned model follows a tuned
    one -- so the cost does not grow with the length of the run.

    Args:
        path (str): Path to the CSV. Created with a header if absent or empty.
        row (dict): Column name -> value for exactly one model on one
            (dataset, iteration, embedding).
    """
    header = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline='') as csvfile:
            header = next(csv.reader(csvfile), None)

    if header is None:
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row), restval='')
            writer.writeheader()
            writer.writerow(row)
        return

    added = [name for name in row if name not in header]
    if not added:
        # restval covers a row that is MISSING a column the header has, which is the
        # other half of the same problem: without it, csv.writer would emit the row's
        # values positionally and silently misalign every column after the gap.
        with open(path, 'a', newline='') as csvfile:
            csv.DictWriter(csvfile, fieldnames=header, restval='').writerow(row)
        return

    with open(path, newline='') as csvfile:
        existing = list(csv.DictReader(csvfile))
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header + added, restval='')
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


@hydra.main(config_path=None, config_name='config', version_base='1.1')
def main(args):
    """
    Main function to run the qprofiler. It initializes logging, sets up the environment, and processes datasets.
    The function reads datasets from the specified folder, applies feature encoding, splits the data into training and test sets,
    applies scaling and embeddings, and evaluates the models using various quantum machine learning methods.
    It logs the results and saves them in a structured format for further analysis. 
    The function also handles parallel processing of multiple machine learning methods and datasets.

    Args:
        args (dict): Configuration parameters for the profiler, including dataset paths, model parameters, and evaluation settings.

    Returns:
        None
    """
    beg_time = time.time() 
    log = logging.getLogger(__name__)
    log.info(f"Main program initiated")
    # Validate before touching args, not after. These three log lines used to sit
    # above this call and indexed 'n_jobs', 'model' and 'backend' directly, so a
    # config missing any of them died with a bare KeyError raised from a logging
    # statement -- the exact "error attributed to the wrong thing" that
    # _validate_config exists to replace, and it reported one missing key where
    # the validator reports all of them at once.
    scaler_name = _validate_config(args, log)

    # Authorise TabPFN's weight download before any worker starts, if the model was asked
    # for. The token has to reach the estimator as $TABPFN_TOKEN, and setting it here puts
    # it in the environment that joblib's workers inherit -- doing it inside the worker
    # would mean reading the credentials file once per model per split. Only the *source*
    # is logged; the token itself is never written to the log, which is committed into
    # results directories and pasted into issues.
    if "tabpfn" in args["model"]:
        from qbiocode.utils.tabpfn_account import load_tabpfn_token

        source = load_tabpfn_token(args)
        if source:
            log.info(f"TabPFN token {source}")
        else:
            log.warning(
                "'tabpfn' is in the model list but no API token was found, so its "
                "pretrained weights cannot be downloaded and the model will fail. "
                "Put the key in ~/.config/qbiocode/tabpfn.json as {\"token\": \"...\"}, "
                "set tabpfn_json_path in the config, or export TABPFN_TOKEN."
            )

    log.info(f"The number of ML methods being parallelized is {min(args['n_jobs'], len(args['model']))}")
    log.info(f"Chosen backend for quantum algorithms is: {args['backend']}")
    # Normalize path separators for cross-platform compatibility
    folder_path = args['folder_path'].replace('/', os.sep).replace('\\', os.sep)
    path_to_input = _resolve_input_folder(folder_path)
    if path_to_input is None:
        raise ValueError(
            f"folder_path {args['folder_path']!r} is not a directory. It was looked "
            f"for relative to the current directory ({os.getcwd()!r}), relative to "
            f"the derived checkout root ({dir_home!r}), and under every parent of "
            f"the current directory. Give an absolute path, or run from a directory "
            f"from which the relative path resolves."
        )
    log.info(f"Reading datasets from {path_to_input}")
    if args['file_dataset'] == 'ALL':
        input_files = [file for file in os.listdir(path_to_input) if file.endswith('csv')]
    else:
        input_files = [file for file in os.listdir(path_to_input) if file in args['file_dataset'] and file.endswith('csv')]
    if not input_files:
        # Previously this produced a successful run with no output whatsoever,
        # which reads exactly like a run whose models all silently failed.
        selector = (
            "every .csv file" if args['file_dataset'] == 'ALL'
            else f"file_dataset={args['file_dataset']!r}"
        )
        raise ValueError(
            f"No input datasets matched {selector} in {path_to_input!r}. "
            f"Directory contents: {sorted(os.listdir(path_to_input))[:10]}"
        )

    # need to populate raw data evaluation for each file, so start an empty list
    appended_raw_data_eval = []
    
    # start looping over datasets
    # start count
    file_count = 0 
    for file in sorted(input_files):
        print(f"Processing file: {file}")
        # this is where the seed needs to be set so the splits are consistent
        np.random.seed(args['seed']) 
        algorithm_globals.random_seed = args['q_seed']

        dataset_start_time = time.time()
        summary = {}
        model_results = {}
        summary.update({'Dataset':file})
        model_results.update({'Dataset':file})
        
        # Load data with optional index column support
        if args.get('index_col', False):
            # First column contains row names/IDs
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,', index_col=0)
            log.info(f"Loaded dataset with row names from first column")
        else:
            # Standard loading without index column
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,')
        
        X = rawdata.iloc[:, :-1].to_numpy()
        y = rawdata.iloc[:,-1:].to_numpy()
        y_encoded = feature_encoding(y, feature_encoding='OrdinalEncoder')
        y_encoded = y_encoded.reshape(-1)
        y_encoded = y_encoded.astype(int)
        y_map = dict(zip(y_encoded.astype(str), y.tolist()))
        summary.update({'label_mapping': y_map})
        
        # Binary classification is a hard requirement, not a preference.
        #
        # This used to warn and continue, calling multi-class support "experimental". It is
        # not experimental, it is absent: `modeleval` scores every model with
        # `roc_auc_score(y_test, y_predicted)` and passes no `multi_class`, so any dataset
        # with more than two classes raised
        #
        #     ValueError: multi_class must be in ('ovo', 'ovr')
        #
        # several hundred lines later, from inside the metrics helper, after the embeddings
        # had been computed and the models fitted. The warning therefore bought nothing: it
        # invited the user to proceed into a failure that was certain, and then reported it
        # against a parameter name they had never heard of. Refusing here names the dataset,
        # the class count, and the fact that this is a boundary rather than a bug.
        n_classes = len(np.unique(y_encoded))
        if n_classes != 2:
            raise ValueError(
                f"Dataset {file!r} has {n_classes} classes, and QProfiler supports binary "
                f"classification only. Its label mapping is {y_map}.\n\n"
                f"This is a boundary of the tool rather than a defect: every model is scored "
                f"through qbiocode.evaluation.modeleval, whose AUC calculation is binary-only, "
                f"so a multi-class run cannot produce a result no matter which models are "
                f"selected. Continuing would fail later, after the embeddings and fits had "
                f"already been computed.\n\n"
                f"To proceed, either restrict the dataset to two classes (a one-vs-rest or "
                f"one-vs-one split of the labels above), or drop it from 'file_dataset'."
            )

        # call and run evaluation functions
        df_dataset = pd.DataFrame(X)
        raw_data_eval = evaluate(df_dataset, y_encoded, file)
        appended_raw_data_eval.append(raw_data_eval)

        # create csv file storing the evaluation of the raw, unembedded data
        all_raw_data_evaluation = pd.concat(appended_raw_data_eval)
        all_raw_data_evaluation.to_csv('RawDataEvaluation.csv', index=False)
        
        # log info
        log.info(f"Started processing data set {file}")
        log.info(f"Dataset has {n_classes} classes: {np.unique(y_encoded).tolist()}")
        
        use_stratify = args.get('stratify', [])
        test_size = args['test_size']
        iter = 0
        # makes number of iterations an argument from config
        for iter in range(args['iter']):
        ## run all this in a loop N_times, while leaving the seed fixed above. The train_test_split will change at each iteration, but will be based on the seed.
            iter=iter+1
            # track iteration time
            iter_start_time = time.time()
            
            # Apply stratification based on config
            # stratify can be: ['y'], ['Y'], or empty list/None for no stratification
            # Distinct-but-reproducible split per iteration: random_state = seed + iter makes
            # every split different from the others, yet deterministic across reruns and
            # independent of any other RNG consumers (embeddings etc.) that run before it.
            split_seed = args['seed'] + iter
            if use_stratify and len(use_stratify) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, stratify=y_encoded, test_size=test_size, random_state=split_seed
                )
                log.info(f"Begin processing iteration (split) {iter} of {args['iter']} with stratified sampling")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=split_seed
                )
                log.info(f"Begin processing iteration (split) {iter} of {args['iter']} without stratification")
            # Scale the features: fit one scaler on TRAIN and apply it to TEST (never fit a
            # separate scaler on the test set -- that would use test-set statistics).
            if scaler_name != 'None':
                X_train, X_test = scale_train_test(X_train, X_test, scaling=scaler_name)
        
            # Embed the training data and test data separately
            for embed in args['embeddings']:
                if embed == 'none':
                    log.info(f"No feature reduction (embedding) applied in this iteration")
                else:
                    log.info(f"Feature reduction (embedding) applied with {embed}")    
                X_train_emb, X_test_emb = get_embeddings(
                    embed, X_train, X_test,
                    n_neighbors=args.get("n_neighbors", 30),
                    n_components=args["n_components"],
                    method=None,
                    quvine_args=args.get("quvine_args", {}),
                )
                summary.update({'embeddings': embed})
                model_results.update({'embeddings': embed})
                
                # TODO: move PQK here as an embedding?

                # call and run evalution functions again if data is embedded, save outputs in the log file
                df_dataset = pd.DataFrame(X_train_emb)
                evaluate_data = evaluate(df_dataset, y_train, file)
                evaluate_data_listofdict = evaluate_data.to_dict(orient='records')
                evaluate_data_dict = {k: v for d in evaluate_data_listofdict for k, v in d.items()}
                # print(evaluate_data_dict)
                model_results.update(evaluate_data_dict)
                #log.info(f"\nThe characteristics of the embedding train dataset are: \n{evaluate_data}")
                summary.update({'iteration': iter})
                model_results.update({'iteration': iter})
                data_key = '_'.join( [re.sub( r'\..*', '', file ), embed, str(args["n_components"]), str(iter)])
                summary.update(model_run(X_train_emb, X_test_emb, y_train, y_test, data_key, args))
                # print(summary)
                # Snapshot the per-(dataset, iteration, embedding) part ONCE, then build
                # each model's row from a fresh copy of it. Merging each model's results
                # into the shared `model_results` instead -- which is what this did --
                # let one model's columns persist into the next model's row: with
                # `grid_search: True` and `tune_quantum: False`, pqk's row carried the
                # preceding model's `BestParams_Tuned` value, so a naive-Bayes
                # `var_smoothing` was reported as pqk's tuned hyperparameters. The rows
                # are independent observations, so they must be built independently.
                row_base = dict(model_results)
                for outerkey, outervalue in summary.items():
                    if outerkey.startswith("results_"):
                        _append_model_row(
                            'ModelResults.csv', {**row_base, **outervalue[0]}
                        )
                # Read existing summary data from the file, if any
                try:
                    with open("results.pkl", "rb") as pklfile:
                        results = pickle.load(pklfile)
                except FileNotFoundError:
                    results = []
                # #Append the list with new summary data
                results.append(summary)
                # Save summary data
                with open('results.pkl', 'wb') as pklfile:
                    pickle.dump(results, pklfile)
            iter_run_time = time.time() - iter_start_time
            
        # start logging times
            log.info(f"The run time for iteration (split) {iter} is: {iter_run_time}")
            
        file_count += 1
        dataset_run_time = time.time() - dataset_start_time
        log.info(f"The total run time for data set {file} is: \n{dataset_run_time}")
        log.info(f"Program has processed {file_count} out of {len(input_files)} data sets")
        log.info(f"Program has {len(input_files)-file_count} data sets left to process")
    
    # log total run time of entire job
    total_run_time = time.time() - beg_time
    log.info(f"\nThe total run time of program is: \n{total_run_time}")

if __name__ == "__main__":
    main()

