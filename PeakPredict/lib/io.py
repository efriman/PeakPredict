# -*- coding: utf-8 -*-
import io
import csv
import pandas as pd
import bioframe
import numpy as np
import logging
import gzip

def is_gz_file(filepath):
    with open(filepath, "rb") as test_f:
        return test_f.read(2) == b"\x1f\x8b"

def sniff_for_header(file, sep="\t", comment="#"):
    """
    Warning: reads the entire file into a StringIO buffer!
    """
    if isinstance(file, str):
        if is_gz_file(file):
            with gzip.open(file, "rt") as f:
                buf = io.StringIO(f.read())
        else:
            with open(file, "r") as f:
                buf = io.StringIO(f.read())
    else:
        buf = io.StringIO(file.read())

    sample_lines = []
    for line in buf:
        if not line.startswith(comment):
            sample_lines.append(line)
            break
    for _ in range(10):
        sample_lines.append(buf.readline())
    buf.seek(0)

    has_header = csv.Sniffer().has_header("\n".join(sample_lines))
    if has_header:
        names = sample_lines[0].strip().split(sep)
    else:
        names = None
    
    ncols = len(sample_lines[0].strip().split(sep))

    return buf, names, ncols

def load_bed(bed_file, 
             schema="bed3", 
             dtypes={"chrom": str, "start": np.int64, "end": np.int64,},
             ):
    buf, names, ncols = sniff_for_header(bed_file)
    if names is not None:
        if set(['chrom', 'start', 'end']).issubset(set(names)):
            features = pd.read_table(buf, dtype=dtypes)
        else:
            raise ValueError("bed file needs to have a header containing chrom, start, and end or no header")
    else:
        features = bioframe.read_table(buf, schema=schema, index_col=False, dtype=dtypes)

    return features
    

