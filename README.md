# overlap_peak_tables
Simple script to count the number of overlapping or closest features between one and multiple bed files.
It can also run models to make predictions of a feature based on these overlaps.

## Installation
`git clone https://github.com/efriman/overlap_peak_tables.git`

`cd overlap_peak_tables`

`pip install .`

## Usage
`overlap_peak_tables base_bed.bed --overlap_features feature1.bed feature2.bed [...] --output outfile.bed [OPTIONS]`

overlap_peak_tables takes a base bed which you want to overlap multiple other bedfiles with (`--overlap_features`). BED files need to either be without headers or contain at least the headers `chrom`, `start`, and `end`. By default, it quantifies the number of peaks overlapping each peak in the base_bed (0/1/2 etc). If you prefer you can set `--boolean_output` to get True/False overlaps. You can also run it with `--closest` to get the number of peaks in vicinity of the peak (parameters for this are `--k`, `--mindist`, and `--maxdist`).

To run predictions based on the peak overlaps, set `--predict_column` to the column in the base_bed you want to predict. Other options for the prediction are `--model`, `--test_size`, `--seed`, and `--plot_size`. By default, the prediction is done using LogisticRegression with `test_size` 0.3 (30%/70% test/training split) without a set seed. Outputs from predictions are:

-Correlation matrix between predictors and feature to be predicted

-Table including predictions for the test data

-Confusion matrix showing true/predicted groups in the test data

-Feature importance graph (For some models, such as GaussianNB, this is done using permutation test, which can take time)


Available models are the following (from scikit-learn, https://scikit-learn.org/):

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


## Example inputs and output
`overlap_peak_tables base_bed.bed --overlap_features overlap_features.bed --output outfile`

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

