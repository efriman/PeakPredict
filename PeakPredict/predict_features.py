#!/usr/bin/env python3
import pandas as pd
import bioframe
import numpy as np
from PeakPredict.PeakPredict_functions import predict_features
import logging
import warnings
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings(action="ignore", message=".*tight_layout.*")
warnings.filterwarnings(action="ignore", message=".*Tight layout.*")
warnings.filterwarnings(action="ignore", message=r".*index_col*")
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(format="%(message)s", level="INFO")


def parse_args_predict_features():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_table",
        type=str,
        help="Tab-separated table containing column to predict and columns to use for prediction",
    )
    parser.add_argument(
        "--predict_column",
        "--predict-column",
        type=str,
        required=True,
        help="""The name of the column to predict""",
    )
    parser.add_argument(
        "--predictor_columns",
        "--predictor-columns",
        type=str,
        nargs="+",
        required=True,
        help="""The name of the columns to use for the prediction""",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        required=False,
        help="""Directory to save in. Defaults to current""",
    )
    parser.add_argument(
        "--outname",
        type=str,
        required=True,
        help="""Prefix for output files. This should NOT be the path to a directory (set with --outdir)""",
    )

    parser.add_argument(
        "--column_type",
        type=str,
        required=False,
        default=False,
        help="""Specify if predict_column contains categorical or numerical values. By default, auto detects""",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="LogisticRegression",
        help="""The name of the model used for prediction. Available models are:
        LogisticRegression
        SVC
        GaussianNB
        MultinomialNB
        SGDClassifier
        KNeighborsClassifier
        DecisionTreeClassifier
        RandomForestClassifier
        GradientBoostingClassifier
        LinearRegression
        SGDRegressor
        KernelRidge
        ElasticNet
        BayesianRidge
        GradientBoostingRegressor
        SVR
        """,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=False,
        default=0.3,
        help="""The fraction of dataset to be split for testing (and the rest for training)""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="""Seed used for random splitting data for prediction. Set for reproducibility""",
    )
    parser.add_argument(
        "--plot_size",
        type=int,
        default=1,
        required=False,
        help="""Relative size of plots. Adjust if they don't look right (too many/few features)""",
    )

    return parser


def main():
    parser = parse_args_predict_features()
    args = parser.parse_args()

    logging.debug(args)

    input_table = pd.read_table(args.input_table)

    if args.column_type and not set([args.column_type]).issubset(
        set(["categorical", "numerical"])
    ):
        warnings.warn(
            "--column type must be either 'categorical' or 'numerical', auto detecting instead"
        )
        args.column_type = False

    if args.seed:
        random_state = args.seed
    else:
        random_state = None

    corr_matrix, predictions, feature_importance = predict_features(
        input_table,
        predict_column=args.predict_column,
        predictor_columns=args.predictor_columns,
        model=args.model,
        test_size=args.test_size,
        random_state=random_state,
        cat_or_num=args.column_type,
    )

    g = sns.clustermap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        yticklabels=True,
        xticklabels=True,
        figsize=(args.plot_size, args.plot_size),
    )
    g.savefig(f"{args.outdir}/{args.outname}_corr_features.png", dpi=100)
    logging.debug(
        f"Saved predictor correlations as {args.outdir}/{args.outname}_corr_features.png"
    )

    predictions.to_csv(
        f"{args.outdir}/{args.outname}_predict_{args.predict_column}_{args.model}.tsv",
        sep="\t",
        index=False,
        header=True,
    )
    logging.debug(
        f"Saved predictions of test data as {args.outdir}/{args.outname}_predict_{args.predict_column}_{args.model}.tsv"
    )

    ConfusionMatrixDisplay.from_predictions(
        predictions[args.predict_column], predictions[f"{args.predict_column}_pred"]
    )
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(
        f"{args.outdir}/{args.outname}_confusion_matrix_{args.predict_column}_{args.model}.png",
        dpi=300,
        bbox_inches="tight",
    )
    logging.debug(
        f"Saved confusion matrix as {args.outdir}/{args.outname}_confusion_matrix_{args.predict_column}_{args.model}.png"
    )

    feature_importance.to_csv(
        f"{args.outdir}/{args.outname}_{args.model}_feature_importance.tsv",
        sep="\t",
        index=False,
    )
    plt.figure(figsize=(args.plot_size, args.plot_size))
    sns.barplot(
        data=feature_importance,
        x="Feature",
        y="Importance",
        color="skyblue",
        edgecolor="black",
    )
    plt.xticks(rotation=45, ha="right")
    plt.savefig(
        f"{args.outdir}/{args.outname}_{args.model}_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    logging.debug(
        f"Saved feature importance as {args.outdir}/{args.outname}_{args.model}_feature_importance.tsv and {args.outdir}/{args.outname}_{args.model}_feature_importance.png"
    )


if __name__ == "__main__":
    main()
