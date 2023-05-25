#!/usr/bin/env python3
import pandas as pd
import bioframe
import numpy as np
from .lib.utils import sniff_for_header
import logging
import warnings
import argparse

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
    return features
    
def count_overlaps(base_features, overlap_bed_file, boolean_output=False):
    if overlap_bed_file in base_features.columns:
        raise ValueError(f"base peaks already contains a column with name {overlap_bed_file}")
    overlap_peaks = load_bed(overlap_bed_file)
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

def count_closest(base_features, overlap_bed_file, k=100, mindist=1, maxdist=1_000_000):
    if overlap_bed_file in base_features.columns:
        raise ValueError(f"base peaks already contains a column with name {overlap_bed_file}")
    overlap_peaks = load_bed(overlap_bed_file)
    ignore_overlaps = False if maxdist==0 else True
    closest = bioframe.closest(base_features, overlap_peaks, k=k, ignore_overlaps=ignore_overlaps)
    closest = closest.loc[(closest["distance"] <= maxdist) &
                          (closest["distance"] >= mindist),:]
    count_closest = closest.groupby("coords").size().reset_index()
    count_closest.columns = ["coords", overlap_bed_file]
    features = pd.merge(base_features, count_closest, on="coords", how="left").fillna(0)
    features[overlap_bed_file] = features[overlap_bed_file].astype(int)
    return features
    
    
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
        type=str,
        nargs="+",
        required=True,
        help="""bed files you want to check overlap with""",
    )
    parser.add_argument(
        "--outname",
        "--o",
        type=str,
        required=True,
        help="""Where to save output file""",
    )
    parser.add_argument(
        "--boolean_output",
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

    return parser

def main():
    parser = parse_args_overlap_count_tables()
    args = parser.parse_args()

    logging.info(args)
    
    schema = "bed3"
    dtypes = {"chrom": str,
              "start": np.int64,
              "end": np.int64,}
    
    base_peaks = load_bed(args.base_bed, 
                          schema=schema,
                          dtypes=dtypes,
                          coord_column=True)
    
    if args.closest:
        for overlap_feature in args.overlap_features:
            logging.info(f"Counting the number of overlapping features with {overlap_feature} within {args.mindist} and {args.maxdist} bp (up to {args.k} allowed, change with --k)")
            base_peaks = count_closest(base_peaks, overlap_feature, k=args.k, mindist=args.mindist, maxdist=args.maxdist)
    else:
        for overlap_feature in args.overlap_features:
            logging.info(f"Checking overlaps with {overlap_feature}")
            base_peaks = count_overlaps(base_peaks, overlap_feature, boolean_output=args.boolean_output)
    
    base_peaks.drop(columns="coords").to_csv(args.outname, sep="\t", index=False, header=True)
    
    logging.info(f"Saved output file as {args.outname}")


if __name__ == '__main__':
    main()