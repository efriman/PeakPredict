# -*- coding: utf-8 -*-
import os
import pandas as pd
import bioframe
import numpy as np
import logging
import warnings
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.inspection import permutation_importance
import sklearn
from sklearn import *
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(format="%(message)s", level="INFO")


def check_chr_naming(peaks1, peaks2, bedpe=False):
    if bedpe:
        return np.any(peaks1["chrom1"].str.contains("chr")) != np.any(
            peaks2["chrom"].str.contains("chr")
        )
    else:
        return np.any(peaks1["chrom"].str.contains("chr")) != np.any(
            peaks2["chrom"].str.contains("chr")
        )


def count_overlaps(
    base_peaks, overlap_peaks, overlap_column_name, boolean_output=False
):
    overlap_table = base_peaks.copy()
    overlap_table["coords"] = (
        overlap_table["chrom"]
        + "_"
        + overlap_table["start"].astype(str)
        + "_"
        + overlap_table["end"].astype(str)
    )
    overlap = bioframe.overlap(
        overlap_table, overlap_peaks, how="both", suffixes=["", "_"]
    )
    overlap_counts = overlap.groupby("coords").size().reset_index()
    overlap_counts.columns = ["coords", overlap_column_name]
    overlap = pd.merge(overlap_table, overlap_counts, on="coords", how="left").fillna(0)
    overlap[overlap_column_name] = overlap[overlap_column_name].astype(np.uint16)
    overlap = overlap.drop(columns="coords")
    if boolean_output:
        overlap[overlap_column_name] = np.where(
            overlap[overlap_column_name] > 0, True, False
        )
    return overlap


def count_closest(
    base_peaks, overlap_peaks, overlap_column_name, k=100, mindist=0, maxdist=1_000_000
):
    overlap_table = base_peaks.copy()
    overlap_table["coords"] = (
        overlap_table["chrom"]
        + "_"
        + overlap_table["start"].astype(str)
        + "_"
        + overlap_table["end"].astype(str)
    )
    ignore_overlaps = False if mindist == 0 else True
    closest = bioframe.closest(
        overlap_table, overlap_peaks, k=k, ignore_overlaps=ignore_overlaps
    )
    closest = closest.loc[
        (closest["distance"] <= maxdist) & (closest["distance"] >= mindist), :
    ]
    count_closest = closest.groupby("coords").size().reset_index()
    count_closest.columns = ["coords", overlap_column_name]
    closest = pd.merge(overlap_table, count_closest, on="coords", how="left").fillna(0)
    closest[overlap_column_name] = closest[overlap_column_name].astype(np.uint16)
    closest = closest.drop(columns="coords")
    return closest


def count_overlaps_bedpe(
    base_peaks, overlap_peaks, overlap_column_name, boolean_output=False
):
    overlap_combined = base_peaks
    for side in [1, 2]:
        side_peaks = base_peaks[[f"chrom{side}", f"start{side}", f"end{side}"]].copy()
        side_peaks.columns = ["chrom", "start", "end"]
        side_peaks["coords"] = (
            side_peaks["chrom"]
            + "_"
            + side_peaks["start"].astype(str)
            + "_"
            + side_peaks["end"].astype(str)
        )
        side_peaks = side_peaks.drop_duplicates()
        overlap = bioframe.overlap(
            side_peaks, overlap_peaks, how="both", suffixes=["", "_"]
        )
        overlap_counts = overlap.groupby("coords").size().reset_index()
        overlap_counts.columns = ["coords", f"{overlap_column_name}{side}"]
        overlap = pd.merge(side_peaks, overlap_counts, on="coords", how="left").fillna(
            0
        )
        overlap[f"{overlap_column_name}{side}"] = overlap[
            f"{overlap_column_name}{side}"
        ].astype(np.uint16)
        overlap = overlap.drop(columns="coords")
        if boolean_output:
            overlap[f"{overlap_column_name}{side}"] = np.where(
                overlap[f"{overlap_column_name}{side}"] > 0, True, False
            )
        overlap.columns = [
            f"chrom{side}",
            f"start{side}",
            f"end{side}",
            f"{overlap_column_name}{side}",
        ]
        overlap_combined = pd.merge(
            overlap_combined, overlap, on=[f"chrom{side}", f"start{side}", f"end{side}"]
        )
    return overlap_combined


def count_closest_bedpe(
    base_peaks, overlap_peaks, overlap_column_name, k=100, mindist=0, maxdist=1_000_000
):
    ignore_overlaps = False if mindist == 0 else True
    closest_combined = base_peaks
    for side in [1, 2]:
        side_peaks = base_peaks[[f"chrom{side}", f"start{side}", f"end{side}"]].copy()
        side_peaks.columns = ["chrom", "start", "end"]
        side_peaks["coords"] = (
            side_peaks["chrom"]
            + "_"
            + side_peaks["start"].astype(str)
            + "_"
            + side_peaks["end"].astype(str)
        )
        side_peaks = side_peaks.drop_duplicates()

        closest = bioframe.closest(
            side_peaks, overlap_peaks, k=k, ignore_overlaps=ignore_overlaps
        )
        closest = closest.loc[
            (closest["distance"] <= maxdist) & (closest["distance"] >= mindist), :
        ]
        count_closest = closest.groupby("coords").size().reset_index()
        count_closest.columns = ["coords", f"{overlap_column_name}{side}"]
        closest = pd.merge(side_peaks, count_closest, on="coords", how="left").fillna(0)
        closest[f"{overlap_column_name}{side}"] = closest[
            f"{overlap_column_name}{side}"
        ].astype(np.uint16)
        closest = closest.drop(columns="coords")
        closest.columns = [
            f"chrom{side}",
            f"start{side}",
            f"end{side}",
            f"{overlap_column_name}{side}",
        ]
        closest_combined = pd.merge(
            closest_combined, closest, on=[f"chrom{side}", f"start{side}", f"end{side}"]
        )
    return closest_combined


def extract_test_train(
    input_table, predict_column, predictor_columns, test_size=0.3, random_state=None
):
    X = input_table[predictor_columns]
    y = input_table[predict_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    logging.debug(f"Training Data Shape: {X_train.shape}")
    logging.debug(f"Testing Data Shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def extract_feature_importance(model, X, y):
    skip_importance = False
    try:
        coefficients = model.coef_
        avg_importance = np.mean(np.abs(coefficients), axis=0)
    except AttributeError:
        try:
            coefficients = model.feature_importances_
            avg_importance = coefficients
        except AttributeError:
            logging.info(
                "Calculating feature importance using permutation (can be slow, Ctrl+C to skip this step)"
            )
            coefficients = permutation_importance(model, X, y)
            avg_importance = coefficients["importances_mean"]
        except:
            warnings.warn("Can't calculate feature importances using this model")
            skip_importance = True
    if skip_importance:
        pd.DataFrame(
            {"Feature": X.columns, "Importance": np.repeat(np.nan, X.shape[1])}
        )
    else:
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": avg_importance}
        )
        feature_importance = feature_importance.sort_values("Importance")
    return feature_importance


def predict_features(
    input_table,
    predict_column,
    predictor_columns,
    model,
    test_size=0.3,
    random_state=None,
    cat_or_num=False,
    **model_kwargs,
):
    if predict_column not in input_table:
        raise ValueError(f"column {predict_column} doesn't exist in the input table")
    if not (set(predictor_columns).issubset(set(input_table.columns))):
        raise ValueError(
            f"At least one predictor column doesn't exist in the input table"
        )

    predict_table = input_table.loc[:, [predict_column] + predictor_columns]

    X_train, X_test, y_train, y_test = extract_test_train(
        predict_table,
        predict_column,
        predictor_columns,
        test_size=test_size,
        random_state=random_state,
    )

    if cat_or_num == "numerical" or (
        not cat_or_num and (isinstance(y_train, float) or isinstance(y_train, int))
    ):
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        corr_matrix = pd.concat([X_train, y_train], axis=1).corr()
    elif cat_or_num == "categorical" or (
        not cat_or_num and isinstance(y_train, object)
    ):
        y_train = y_train.astype("category")
        y_test = y_test.astype("category")
        logging.debug(
            "Can't correlate features with categorical outcome variable, plotting only predictor correlations"
        )
        corr_matrix = X_train.corr()

    for module in dir(sklearn):
        if model in dir(eval(f"sklearn.{module}")):
            logging.info(f"Predicting {predict_column} using sklearn.{module}.{model}")
            logging.info(f"Using model parameters: {model_kwargs}")
            model = eval(f"sklearn.{module}.{model}")(**model_kwargs)
            break
    if isinstance(model, str):
        raise ValueError(f"{model} doesn't exist in sklearn")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        mcc = matthews_corrcoef(y_test, y_pred)
        logging.info(f"Matthew's correlation coefficient: {mcc}")
    except ValueError:
        warnings.warn("Cannot compute Matthew's correlation coefficient for this type of classification/regression, see https://stackoverflow.com/a/54458777")
    

    predictions = X_test.copy()
    predictions[f"{predict_column}_pred"] = y_pred
    predictions = pd.merge(
        input_table,
        predictions.drop(columns=predictor_columns),
        how="right",
        left_index=True,
        right_index=True,
    )

    feature_importance = extract_feature_importance(model, X_test, y_test)

    return corr_matrix, predictions, feature_importance, model
