# overlap_peak_tables
Simple script to count the number of overlapping or closest features between one and multiple bed files

## Installation
`git clone https://github.com/efriman/overlap_peak_tables.git`

`cd overlap_peak_tables`

`pip install .`

## Usage
`overlap_peak_tables base_bed.bed --overlap_features feature1.bed feature2.bed --output outfile.bed [OPTIONS]`

overlap_peak_tables takes a base bed which you want to overlap multiple other bedfiles with (`--overlap_features`). BED files need to either be without headers or contain at least the headers `chrom`, `start`, and `end`. By default, it quantifies the number of peaks overlapping each peak in the base_bed (0/1/2 etc). If you prefer you can set `--boolean_output` to get True/False overlaps. You can also run it with `--closest` to get the number of peaks in vicinity of the peak (parameters for this are `--k`, `--mindist`, and `--maxdist`).


## Inputs and outputs (examples)
`overlap_peak_tables base_bed.bed --overlap_features overlap_features.bed --output outfile.bed`

base_bed.bed
| chrom  | start | end |
| ------------- | ------------- | ------------- |
| chr1  | 1  | 100 |
| chr1  | 1000  | 2000 |
| chr1  | 5000  | 5300 |

overlap_features.bed
| chrom  | start | end |
| ------------- | ------------- | ------------- |
| chr1  | 5  | 50 |
| chr1  | 5100  | 5500 |

outfile.bed
| chrom  | start | end | overlap_features.bed |
| ------------- | ------------- | ------------- | ------------- |
| chr1  | 1  | 100 | 1 |
| chr1  | 1000  | 2000 | 0 |
| chr1  | 5000  | 5300 | 1 |