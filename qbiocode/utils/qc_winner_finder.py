## function to find datasets where QML methods did better than classical
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def qml_winner(results_df, rawevals_df, output_dir, tag):
    """This function finds data sets where QML was beneficial (higher F1 scores than CML) and create new .csv files
    with the relevant evaluation and performance for these specific datasets, for further analysis.
    It also computes the best results per method across all splits and the best results per dataset.
    It returns two DataFrames: one with the datasets where QML methods outperformed CML methods, and another with the
    evaluation scores for the best QML method for each of these datasets.
    It also saves these DataFrames as .csv files in the specified output directory.

    Args:
        results_df (pandas.DataFrame): Dataset in pandas corresponding to 'ModelResults.csv'
        rawevals_df (pandas.DataFrame): Dataset in pandas corresponding to 'RawDataEvaluation.csv'
    Returns:
        qml_winners (pandas.DataFrame): contais the input datasets for which at least one QML method
                                        performed better than CML. DataFrame contains the scores of all
                                        the methods.
        winner_eval_score (pandas.DataFrame): contains the input datasets, their evaluation, and scores for the
                                            specific qml method that yielded the best score.
    """

    # pass in the ML results
    df = results_df.copy()
    # pull in the raw evaluations
    rawevals = rawevals_df.copy()
    # first, compute mean across all splits
    # model_evaluation.py writes exactly one parameter column, named for the branch
    # that produced it: 'Model_Parameters' with tuning off, 'BestParams_Tuned' with it
    # on -- or 'BestParams_GridSearch', the name that branch used before Optuna
    # replaced the exhaustive grid, which every older ModelResults.csv still carries.
    # The previous if/else assumed the absence of one meant the presence of the other,
    # so any third name became a KeyError raised from inside groupby.
    parameter_columns = [
        name for name in ("Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch")
        if name in df.columns
    ]
    if not parameter_columns:
        raise ValueError(
            "None of the model-parameter columns ('Model_Parameters', "
            "'BestParams_Tuned', 'BestParams_GridSearch') are present in the results "
            f"table, which has {sorted(df.columns)}. Pass the ModelResults.csv written "
            "by QProfiler."
        )
    # Coalesced, not `parameter_columns[0]`. A single run can carry BOTH names now that
    # model_evaluation decides the column per row rather than per run (a tuned classical
    # model reports 'BestParams_Tuned' while an untuned quantum one in the same run
    # reports 'Model_Parameters'). Grouping on whichever name happened to sort first left
    # NaN in that key for every row belonging to the other column -- and pandas' groupby
    # drops NaN keys by default, so those rows would disappear from the winner table
    # silently. Taking the first non-null across the candidates recovers each row's real
    # parameters whichever column holds them.
    parameters = df[parameter_columns].bfill(axis=1).iloc[:, 0]
    df_across_split = (
        df.assign(_parameters=parameters)
        .groupby(["Dataset", "embeddings", "model", "_parameters"])["f1_score"]
        .mean()
        .reset_index()
        .rename(columns={"_parameters": parameter_columns[0]})
    )
    # now, extract the best results per method across embedding and iteration
    df_best = df_across_split.groupby(["Dataset", "model"])["f1_score"].max().reset_index()
    # df_best = df_across_split.groupby(['Dataset', 'model', 'Model_Parameters'])['f1_score'].max().reset_index()
    df_best.to_csv((os.path.join(output_dir, tag + "_best_across_split.csv")), index=False)
    # get summary accross all datasets
    df_best_model_mean = df_best.groupby("model")["f1_score"].mean()
    df_best_model_median = df_best.groupby("model")["f1_score"].median()
    df_best_model_max = df_best.groupby("model")["f1_score"].max()
    df_best_model_std = df_best.groupby("model")["f1_score"].std()
    df_best_permodel_summary = pd.concat(
        [df_best_model_mean, df_best_model_median, df_best_model_max, df_best_model_std], axis=1
    )
    df_best_permodel_summary.columns = [
        "Mean_F1_Score",
        "Median_F1_Score",
        "Max_F1_Score",
        "StandardDev_F1_Score",
    ]
    df_best_permodel_summary.to_csv((os.path.join(output_dir, tag + "_best_permodel_summary.csv")))
    # print(df_best_permodel_summary)

    # extract the best results per dataset
    best_per_dataset = df_best.loc[df_best.groupby("Dataset")["f1_score"].idxmax()]
    # best_per_dataset = df_across_split.loc[df_across_split.groupby('Dataset')['f1_score'].idxmax()]
    # create list of qml methods
    # Matched on the lower-cased first token of the label, not against a fixed list of
    # exact spellings. The list used to be ["QSVC", "QNN", "VQC", "PQK"] compared with
    # `.isin()`, which could never match: every label QProfiler writes is lower case, so
    # this branch was dead on real output and qml_winners.csv came back empty from every
    # genuine quantum run. Two further spellings it could not have matched either way --
    # 'qpl' was absent from the list entirely, and a tuned run is labelled '<name>_opt'
    # ('qpl_opt_<head>' for QPL, which fans out one column per classical head).
    qml_families = {"qsvc", "qnn", "vqc", "pqk", "qpl"}

    def _is_quantum(label):
        return str(label).lower().split("_", 1)[0] in qml_families

    quantum_datasets = best_per_dataset.loc[
        best_per_dataset["model"].map(_is_quantum), "Dataset"
    ]
    qml_winner = df_across_split[df_across_split["Dataset"].isin(quantum_datasets)]
    if not qml_winner.empty:
        bestmethod = qml_winner.groupby("Dataset")["f1_score"].idxmax()
        qc_method_and_score = qml_winner.loc[bestmethod]
        qml_winner.to_csv((os.path.join(output_dir, tag + "_qml_winners.csv")), index=False)
        dataset = list(qml_winner["Dataset"].unique())

        #######
        # now let's find the raw data evaluations for the qml winner data sets
        # this wil produce another csv file that contains scores, evaluation, and qml method
        # for these "qml winners".
        winner_evals = []
        for file in dataset:
            eval = rawevals.loc[rawevals["Dataset"] == file]
            # print(eval)
            winner_evals.append(eval)
        winner_evals_df = pd.concat(winner_evals)
        winner_evals_df.to_csv((os.path.join(output_dir, tag + "_winner_evals.csv")), index=False)
        winner_scores_df = qc_method_and_score.iloc[:, -3:]
        winner_scores_df.to_csv((os.path.join(output_dir, tag + "_winner_score.csv")), index=False)

        print(winner_scores_df)
        winner_eval_score = pd.concat([winner_evals_df, winner_scores_df], axis=1)
        winner_eval_score.to_csv(
            (os.path.join(output_dir, tag + "_winner_eval_score.csv")), index=False
        )  # contains dataset, evaluation, qml method, and  average f1 score
        #######

        # optional print statements
        print("*** The number of qml winners is", len(dataset))
        print("*** The qml winners are:", dataset)

        return qml_winner, winner_eval_score, df_best

    else:
        print("*** QML methods were outperformed by CML methods in all datasets ***")

        return
