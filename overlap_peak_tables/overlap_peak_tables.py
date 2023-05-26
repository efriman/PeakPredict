#!/usr/bin/env python3
import pandas as pd
import bioframe
import numpy as np
from .lib.utils import sniff_for_header
import logging
import warnings
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings(action="ignore", message=r".*index_col*")
logging.basicConfig(format="%(message)s", level='INFO')

def load_bed(bed_file, 
             schema="bed3", 
             dtypes={"chrom": str, "start": np.int64, "end": np.int64,},
             coord_column=False):
    buf, names, ncols = sniff_for_header(bed_file)
    if names is not None:
        if set(['chrom', 'start', 'end']).issubset(set(names)):
            features = pd.read_table(buf, dtype=dtypes)
        else:
            raise ValueError("bed file needs to have a header containing chrom, start, and end or no header")
    else:
        features = bioframe.read_table(buf, schema=schema, index_col=False, dtype=dtypes)
    if coord_column:
        if "coords" in names:
            logging.info("Overwriting column named coords, rename if needed")
        features["coords"] = features["chrom"] + "_" + features["start"].astype(str) + "_" + features["end"].astype(str)
    return features, np.any(features["chrom"].str.contains("chr"))
    
def count_overlaps(base_features, overlap_bed_file, chr_or_no, boolean_output=False):
    if overlap_bed_file in base_features.columns:
        raise ValueError(f"base peaks already contains a column with name {overlap_bed_file}")
    overlap_peaks, chr_feature = load_bed(overlap_bed_file)
    if chr_or_no != chr_feature:
        warnings.warn("Base bed/features have mixed chrososome naming (i.e. using chr or not)")
    overlap = bioframe.overlap(base_features, overlap_peaks, how='both', suffixes=['', '_'])
    if not boolean_output:
        overlap_counts = overlap.groupby("coords").size().reset_index()
        overlap_counts.columns = ["coords", overlap_bed_file]
        features = pd.merge(base_features, overlap_counts, on="coords", how="left").fillna(0)
        features[overlap_bed_file] = features[overlap_bed_file].astype(int)
    else:
        features = base_features
        overlap["coords"] = overlap["chrom"] + "_" + overlap["start"].astype(str) + "_" + overlap["end"].astype(str)
        features[overlap_bed_file] = np.where(features["coords"].isin(list(overlap["coords"])), True, False)
    return features

def count_closest(base_features, overlap_bed_file, chr_or_no, k=100, mindist=1, maxdist=1_000_000):
    if overlap_bed_file in base_features.columns:
        raise ValueError(f"base peaks already contains a column with name {overlap_bed_file}")
    overlap_peaks, chr_feature = load_bed(overlap_bed_file)
    if chr_or_no != chr_feature:
        warnings.warn("Base bed/features have mixed chrososome naming (i.e. using chr or not)")
    ignore_overlaps = False if maxdist==0 else True
    closest = bioframe.closest(base_features, overlap_peaks, k=k, ignore_overlaps=ignore_overlaps)
    closest = closest.loc[(closest["distance"] <= maxdist) &
                          (closest["distance"] >= mindist),:]
    count_closest = closest.groupby("coords").size().reset_index()
    count_closest.columns = ["coords", overlap_bed_file]
    features = pd.merge(base_features, count_closest, on="coords", how="left").fillna(0)
    features[overlap_bed_file] = features[overlap_bed_file].astype(int)
    return features

def extract_test_train(input_table, predict_column, predictors_columns, test_size=0.3, random_state=None):
    X = input_table[predictors_columns]
    y = input_table[predict_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    logging.info(f"Training Data Shape: {X_train.shape}")
    logging.info(f"Testing Data Shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def correlate_features(X, y):
    train_df = pd.concat([X, y], axis=1)
    corr_matrix = train_df.corr()
    return corr_matrix

def model_predict(X_train, X_test, y_train, y_test, model):
    model = model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    logging.info(f"Matthew's correlation coefficient: {mcc}")
    return y_pred, model

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
            logging.info("Calculating feature importance using permutation (can be slow, Ctrl+C to skip this step)")
            coefficients = permutation_importance(model, X, y)
            avg_importance = coefficients["importances_mean"]
        except:
            warnings.warn("Can't calculate feature importances using this model")
            skip_importance = True
    if skip_importance:
        pd.DataFrame({'Feature': X.columns, 'Importance': np.repeat(np.nan, X.shape[1])})
    else:
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=True)
    return feature_importance

    
def parse_args_overlap_count_tables():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "base_bed", 
        type=str, 
        help="bed file you want to overlap other features with"
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
        "--output",
        "--outname",
        "--o",
        type=str,
        required=True,
        help="""Prefix for output files. Output table will be saved as .tsv""",
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
        help="""Whether to count the number of closest peaks within a certain distance instead of just overlapping""",
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
        default=1,
        help="""Minimum distance for the number of closest peaks. Set to 0 to include overlaps""",
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
    parser = parse_args_overlap_count_tables()
    args = parser.parse_args()

    logging.info(args)
    
    schema = "bed3"
    dtypes = {"chrom": str,
              "start": np.int64,
              "end": np.int64,}
    
    base_peaks, chr_or_no = load_bed(args.base_bed, 
                                     schema=schema,
                                     dtypes=dtypes,
                                     coord_column=True)
    
    overlap_table = base_peaks.copy()
    
    if args.closest:
        for overlap_feature in args.overlap_features:
            logging.info(f"Counting the number of overlapping features with {overlap_feature} within {args.mindist} and {args.maxdist} bp (up to {args.k} allowed, change with --k)")
            overlap_table = count_closest(overlap_table, 
                                          overlap_feature, 
                                          chr_or_no=chr_or_no,
                                          k=args.k, 
                                          mindist=args.mindist, 
                                          maxdist=args.maxdist)
    else:
        for overlap_feature in args.overlap_features:
            logging.info(f"Checking overlaps with {overlap_feature}")
            overlap_table = count_overlaps(overlap_table, 
                                           overlap_feature,
                                           chr_or_no=chr_or_no,
                                           boolean_output=args.boolean_output)

    overlap_table.drop(columns="coords").to_csv(f"{args.output}.tsv", sep="\t", index=False, header=True)
    
    logging.info(f"Saved overlap table as {args.output}.tsv")
    
    if args.predict_column is not None:
        if args.predict_column not in overlap_table:
            raise ValueError(f"column {args.predict_column} doesn't exist in the base bed")
        if args.seed:
            random_state = args.seed
        else:
            random_state = None
        predictors_columns = [col for col in list(overlap_table.columns) if col not in (base_peaks.columns)]
        X_train, X_test, y_train, y_test = extract_test_train(overlap_table, 
                                                              args.predict_column, 
                                                              predictors_columns, 
                                                              test_size=args.test_size, 
                                                              random_state=random_state)
        logging.info(f"Plotting correlation matrix as {args.output}.png")
        corr_matrix = correlate_features(X_train, y_train)
        g = sns.clustermap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, 
                           yticklabels=True, xticklabels=True,
                           figsize=(args.plot_size,args.plot_size))
        g.savefig(f"corr_features_{args.output}.png", dpi=100)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import SGDRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import BayesianRidge
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.svm import SVR
        
        model_dict = {"LogisticRegression":LogisticRegression,
                "SVC":SVC,
                "GaussianNB":GaussianNB,
                "MultinomialNB":MultinomialNB,
                "SGDClassifier":SGDClassifier,
                "KNeighborsClassifier":KNeighborsClassifier,
                "DecisionTreeClassifier":DecisionTreeClassifier,
                "RandomForestClassifier":RandomForestClassifier,
                "GradientBoostingClassifier":GradientBoostingClassifier,
                "LinearRegression":LinearRegression,
                "SGDRegressor":SGDRegressor,
                "KernelRidge":KernelRidge,
                "ElasticNet":ElasticNet,
                "BayesianRidge":BayesianRidge,
                "GradientBoostingRegressor":GradientBoostingRegressor,
                "SVR":SVR,}

        if args.model not in model_dict.keys():
            raise ValueError(f"{args.model} is not available")

        logging.info(f"Predicting {args.predict_column} based on overlap counts using {args.model}")
        y_pred, model_fit = model_predict(X_train, X_test, y_train, y_test, model=model_dict[args.model])
        predictions = X_test.join(base_peaks).reset_index(drop=True)
        predictions[f"{args.predict_column}_pred_{args.model}"] = y_pred 
        predictions.to_csv(f"{args.output}_predict_{args.predict_column}_{args.model}.tsv", sep="\t", index=False, header=True)
        logging.info(f"Saved predictions of test data as {args.output}_predict_{args.predict_column}_{args.model}.tsv")
        
        logging.info(f"Plotting confusion matrix as {args.output}_confusion_matrix_{args.predict_column}_{args.model}.png")
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.tight_layout()
        plt.savefig(f"{args.output}_confusion_matrix_{args.predict_column}_{args.model}.png", dpi=300, bbox_inches="tight")
        
        logging.info(f"Plotting feature importance as {args.output}_{args.model}_feature_importance.png")
        feature_importance = extract_feature_importance(model_fit, X_test, y_test)
        plt.figure(figsize=(args.plot_size,args.plot_size))
        sns.barplot(data=feature_importance, x='Feature', y='Importance', color="skyblue", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.savefig(f"{args.output}_{args.model}_feature_importance.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    main()