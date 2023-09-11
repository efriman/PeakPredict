#!/usr/bin/env python3
import pandas as pd
import bioframe
import numpy as np
from PeakPredict.lib.io import load_bed
from PeakPredict.PeakPredict_functions import *
import logging
import warnings
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import json
import shap

warnings.filterwarnings(action="ignore", message=".*tight_layout.*")
warnings.filterwarnings(action="ignore", message=".*Tight layout.*")
warnings.filterwarnings(action="ignore", message=r".*index_col*")
warnings.filterwarnings(action="ignore", message=r".*No data for colormapping*")
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(format="%(message)s", level="INFO")


def parse_args_overlap_peaks():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "base_bed", type=str, help="bed file you want to overlap other features with"
    )
    parser.add_argument(
        "--overlap_features",
        "--overlap-features",
        type=str,
        nargs="+",
        required=True,
        help="""bed files you want to check overlap with""",
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
        "--bedpe",
        action="store_true",
        default=False,
        required=False,
        help="""Specify if you have a bedpe file as base_bed. Will count overlap on each side separately""",
    )
    parser.add_argument(
        "--boolean_output",
        "--boolean-output",
        action="store_true",
        default=False,
        required=False,
        help="""Returns overlap outputs as True/False instead of counts. Only for when not using closest""",
    )
    parser.add_argument(
        "--closest",
        action="store_true",
        default=False,
        required=False,
        help="""Whether to count the number of closest peaks within a certain distance instead of overlapping""",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=100,
        help="""Maximum number regions in proximity to look for""",
    )
    parser.add_argument(
        "--mindist",
        type=int,
        required=False,
        default=0,
        help="""Minimum distance for the number of closest peaks. Set to >0 to avoid counting overlaps""",
    )
    parser.add_argument(
        "--maxdist",
        type=int,
        required=False,
        default=1_000_000,
        help="""Maximum distance for the number of closest peaks""",
    )
    parser.add_argument(
        "--predict_column",
        "--predict-column",
        type=str,
        required=False,
        default=None,
        help="""The name of the column to predict based on features used in the overlap""",
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
        help="""The name of the model used for prediction. All models from sklearn are available
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
        "--balance",
        action="store_true",
        default=False,
        required=False,
        help="""Specify if you want to downsample the different categories to equal the smallest""",
    )
    parser.add_argument(
        "--maximum_per_category",
        type=int,
        default=0,
        help="""Specify >0 if you want to downsample the different categories to this number (or smaller)""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="""Seed used for random splitting data for prediction. Set for reproducibility""",
    )
    parser.add_argument(
        "--model_args",
        "--model-args",
        type=json.loads,
        default={},
        help="""Additional arguments to use in the prediction model. Supply as dictionary,
        e.g. '{"n_estimators": 200, "criterion": "entropy"} """,
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        default=False,
        required=False,
        help="""Specify if you want to generate SHAP value plots""",
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
    parser = parse_args_overlap_peaks()
    args = parser.parse_args()

    logging.debug(args)

    if args.bedpe:
        schema = "bedpe"
        dtypes = {
            "chrom1": str,
            "start1": np.int32,
            "end1": np.int32,
            "chrom2": str,
            "start2": np.int32,
            "end2": np.int32,
        }
    else:
        schema = "bed3"
        dtypes = {
            "chrom": str,
            "start": np.int32,
            "end": np.int32,
        }

    base_peaks = load_bed(args.base_bed, schema=schema, dtypes=dtypes)

    if "coords" in base_peaks.columns:
        warnings.warn("column named coords will be removed")

    overlap_table = base_peaks.copy()

    if args.closest:
        logging.info(
            f"Counting the number of peaks within {args.mindist} and {args.maxdist} bp (up to {args.k} allowed, change with --k)"
        )
        for overlap_feature in args.overlap_features:
            logging.info(overlap_feature)
            if overlap_feature in overlap_table.columns:
                raise ValueError(
                    f"base peaks already contains a column with name {overlap_feature}"
                )
            else:
                overlap_peaks = load_bed(
                    overlap_feature,
                    schema="bed3",
                    dtypes={
                        "chrom": str,
                        "start": np.int32,
                        "end": np.int32,
                    },
                )
                if overlap_peaks.empty:
                    logging.info(f"Something wrong with the format of {overlap_feature}, skipping")
                    continue
                if check_chr_naming(overlap_table, overlap_peaks, bedpe=args.bedpe):
                    warnings.warn(
                        "The peak files have inconsistent naming with regards to using 'chr'"
                    )
                if args.bedpe:
                    overlap_table = count_closest_bedpe(
                        overlap_table,
                        overlap_peaks,
                        overlap_feature,
                        k=args.k,
                        mindist=args.mindist,
                        maxdist=args.maxdist,
                    )
                else:
                    overlap_table = count_closest(
                        overlap_table,
                        overlap_peaks,
                        overlap_feature,
                        k=args.k,
                        mindist=args.mindist,
                        maxdist=args.maxdist,
                    )
    else:
        logging.info(f"Counting overlaps")
        for overlap_feature in args.overlap_features:
            logging.info(overlap_feature)
            if overlap_feature in overlap_table.columns:
                raise ValueError(
                    f"base peaks already contains a column with name {overlap_feature}"
                )
            else:
                overlap_peaks = load_bed(
                    overlap_feature,
                    schema="bed3",
                    dtypes={
                        "chrom": str,
                        "start": np.int32,
                        "end": np.int32,
                    },
                )
                if overlap_peaks.empty:
                    logging.info(f"Something wrong with the format of {overlap_feature}, skipping")
                    continue
                if check_chr_naming(overlap_table, overlap_peaks, bedpe=args.bedpe):
                    warnings.warn(
                        "The peak files have inconsistent naming with regards to using 'chr'"
                    )
                if args.bedpe:
                    overlap_table = count_overlaps_bedpe(
                        overlap_table,
                        overlap_peaks,
                        overlap_feature,
                        boolean_output=args.boolean_output,
                    )
                else:
                    overlap_table = count_overlaps(
                        overlap_table,
                        overlap_peaks,
                        overlap_feature,
                        boolean_output=args.boolean_output,
                    )

    overlap_table.to_csv(
        f"{args.outdir}/{args.outname}.tsv", sep="\t", index=False, header=True
    )
    logging.info(f"Saved overlap table as {args.outdir}/{args.outname}.tsv")

    if args.predict_column is not None:
        if args.balance:
            if args.maximum_per_category > 0:
                logging.info("balance supersedes maximum_per_category")
            smallest = min(
                overlap_table.groupby(args.predict_column).size().reset_index()[0]
            )
            logging.info(f"Downsampling to {smallest} regions per group")
            input_table = (
                overlap_table.groupby(args.predict_column)
                .sample(smallest)
                .reset_index(drop=True)
            )
        elif args.maximum_per_category > 0:
            logging.info(
                f"Downsampling to {args.maximum_per_category} regions per group"
            )
            input_table = (
                overlap_table.groupby(args.predict_column)
                .sample(args.maximum_per_category)
                .reset_index(drop=True)
            )
        else:
            input_table = overlap_table

        predictor_columns = [
            col
            for col in list(overlap_table.columns)
            if col not in (base_peaks.columns)
        ]

        if args.column_type and not set([args.column_type]).issubset(
            set(["categorical", "numerical"])
        ):
            warnings.warn(
                "--column_type must be either 'categorical' or 'numerical', auto detecting instead"
            )
            args.column_type = False

        if args.seed:
            random_state = args.seed
        else:
            random_state = None

        corr_matrix, predictions, feature_importance, model = predict_features(
            input_table,
            predict_column=args.predict_column,
            predictor_columns=predictor_columns,
            model=args.model,
            test_size=args.test_size,
            random_state=random_state,
            cat_or_num=args.column_type,
            **args.model_args,
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
        logging.info(
            f"Saved predictor correlations as {args.outdir}/{args.outname}_corr_features.png"
        )

        predictions.to_csv(
            f"{args.outdir}/{args.outname}_predict_{args.predict_column}_{args.model}.tsv",
            sep="\t",
            index=False,
            header=True,
        )
        logging.info(
            f"Saved predictions of test data as {args.outdir}/{args.outname}_predict_{args.predict_column}_{args.model}.tsv"
        )
        try:
            ConfusionMatrixDisplay.from_predictions(
                predictions[args.predict_column],
                predictions[f"{args.predict_column}_pred"],
            )
            plt.tight_layout()
            plt.xticks(rotation=90)
            plt.savefig(
                f"{args.outdir}/{args.outname}_confusion_matrix_{args.predict_column}_{args.model}.png",
                dpi=300,
                bbox_inches="tight",
            )
            logging.info(
                f"Saved confusion matrix as {args.outdir}/{args.outname}_confusion_matrix_{args.predict_column}_{args.model}.png"
            )
        except ValueError:
            warnings.warn(
                "Cannot generate Confusion Matrix for this type of classification/regression, see https://stackoverflow.com/a/54458777"
            )

        feature_importance.to_csv(
            f"{args.outdir}/{args.outname}_feature_importance_{args.model}.tsv",
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
            f"{args.outdir}/{args.outname}_feature_importance_{args.model}.png",
            dpi=300,
            bbox_inches="tight",
        )
        logging.info(
            f"Saved feature importance as {args.outdir}/{args.outname}_feature_importance_{args.model}.tsv and {args.outdir}/{args.outname}_feature_importance_{args.model}.png"
        )

        if args.shap:
            logging.info(f"Calculating SHAP values (can be slow for big datasets)")
            plot_size = (
                None if args.plot_size == 1 else (args.plot_size, args.plot_size)
            )
            shap_values = shap.Explainer(model).shap_values(
                predictions[predictor_columns]
            )
            plt.figure()
            shap.summary_plot(
                shap_values,
                predictions[predictor_columns],
                class_names=model.classes_,
                show=False,
                plot_type="bar",
                plot_size=plot_size,
            )
            plt.legend(loc=(1.04, 0))
            plt.savefig(
                f"{args.outdir}/{args.outname}_SHAP_{args.model}.png",
                dpi=300,
                bbox_inches="tight",
            )
            logging.info(
                f"Saved SHAP values as {args.outdir}/{args.outname}_SHAP_{args.model}.png"
            )
            for i in range(len(shap_values)):
                name = model.classes_[i]
                plt.figure()
                shap.summary_plot(
                    shap_values[i],
                    predictions[predictor_columns],
                    show=False,
                    plot_size=plot_size,
                )
                plt.title(label=name)
                plt.savefig(
                    f"{args.outdir}/{args.outname}_SHAP_{name}_{args.model}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info(
                    f"Saved SHAP values as {args.outdir}/{args.outname}_SHAP_{name}_{args.model}.png"
                )


if __name__ == "__main__":
    main()
