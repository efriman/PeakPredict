# PeakPredict
> Peak overlaps and feature predictions

Command line interface to count the number of overlapping or closest features between one and multiple bed files and run models to make predictions of features based on these overlaps.

## Installation
`git clone https://github.com/efriman/PeakPredict.git`

`cd PeakPredict`

`pip install .`

## Usage
`overlap_peaks base_bed.bed --overlap_features feature1.bed feature2.bed [...] --outname outfile [OPTIONS]`

`overlap_peaks` takes a base BED (or BEDPE, specify `--bedpe`) which you want to overlap multiple other bed files with (`--overlap_features`). Files need to either be without headers or contain at least the headers `chrom`, `start`, and `end`. By default, it quantifies the number of peaks overlapping each peak in the base_bed (0/1/2 etc). If you prefer you can set `--boolean_output` to get True/False overlaps. You can also run it with `--closest` to get the number of peaks in vicinity of the peak (parameters for this are `--k`, `--mindist`, and `--maxdist`).

To run predictions based on the peak overlaps, set `--predict_column` to the column in the base_bed you want to predict. Other options for the prediction are `--column_type`, `--model`, `--test_size`, `--seed`, and `--plot_size`. Predictions can also be done on any input table using

`predict_features input_table.tsv --predict_column predict_this --predictor_columns [...] --outname outfile [OPTIONS]`

Outputs from predictions are:

-Correlation matrix between predictors and feature to be predicted

-Table including predictions for the test data

-Confusion matrix showing true/predicted groups in the test data

-Feature importance graph (For some models, this is done using permutation test, which can take time)

All models from [scikit-learn](https://scikit-learn.org/) are available to use. Some models might require categorical or numerical values specifically.


## Example inputs and output
`overlap_peaks base_bed.bed --overlap_features overlap_features1.bed overlap_features2.bed --outname outfile`

base_bed.bed
| chrom  | start | end |
| ------------- | ------------- | ------------- |
| chr1  | 1  | 100 |
| chr1  | 1000  | 2000 |
| chr1  | 5000  | 5300 |

overlap_features1.bed
| chrom  | start | end |
| ------------- | ------------- | ------------- |
| chr1  | 5  | 50 |
| chr1  | 5100  | 5500 |

overlap_features2.bed
| chrom  | start | end |
| ------------- | ------------- | ------------- |
| chr1  | 10  | 20 |
| chr1  | 1200  | 1300 |

outfile.tsv
| chrom  | start | end | overlap_features1.bed | overlap_features2.bed |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| chr1  | 1  | 100 | 1 | 1 |
| chr1  | 1000  | 2000 | 0 | 1 |
| chr1  | 5000  | 5300 | 1 | 0 |

## Walkthrough
An walkthrough of basic examples can be found here: [Walkthrough](https://github.com/efriman/PeakPredict/blob/main/docs/Walkthrough_examples.ipynb)
