# -*- coding: utf-8 -*-
import io
import csv

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