# Statistics

When analyzing a data set, descriptive statistics are calculated for the whole data set and for each split separately.
They are saved in subdirectories `/statistics/` of the main CaTabRa folder and the respective split-subdirectories in
the evaluation folders.

Three files are generated: `correlation.xlsx`, `statistics_numeric.xlsx` and `statistics_non_numeric.xlsx`.

## `correlation.xlsx`

This file contains pairwise Pearson correlation values of all numerical input features.

Note that the file is not generated if the number of numerical features exceeds a certain pre-defined limit.

## `statistics_numeric.xlsx`

This file contains statistics of all numerical features, including

* `count`: number of **non-NA** values,
* `mean`: mean of all non-NA values,
* `std`: standard deviation of all non-NA values,
* `min`: minimum observed non-NA value,
* `25%`: first quartile of all non-NA values,
* `50%`: median of all non-NA values,
* `75%`: third quartile of all non-NA values,
* `max`: maximum observed non-NA value.

In case of classification tasks (binary, multiclass, multilabel), the above statistics are also calculated for each
class individually and included as additional sheets in the Excel file. The statistic `mann_whitney_u` is calculated
in addition to the statistics listed above, representing the p-value of the two-sided
[Mann-Whitney *U* test](https://en.wikipedia.org/wiki/Mann-Whitney_U_test) when comparing the feature values of each
class to the values of all other classes ("one-vs-rest" schema). The Mann-Whitney *U* test is a special case of the
Kruskal-Wallis *H* test, which allows to compare more than two groups (or *samples*) simultaneously; they are
equivalent when comparing two groups. Also note that the Mann-Whitney *U* test is symmetric between the two groups to
compare, meaning that the p-values will be equal for both classes if there are only two of them (binary- and multilabel
classification without unlabeled samples). In any case, unlabeled samples are treated as a class of their own.

## `statistics_non_numeric.xlsx`

This file contains statistics of all non-numerical (categorical, boolean) features, including

* `count`: total number of occurrences of each category,
* `%`: relative frequency of each category, defined as `count / N` (where `N` denotes number of samples).

If applicable, NA categories are listed separately.

In case of classification tasks (binary, multiclass, multilabel), the above statistics are also calculated for each
class individually and included as additional sheets in the Excel file. The columns in these sheets are not called
`count` and `%`, but are prefixed with the class name for every class. Relative frequencies are divided by the total
number of samples in each class, such that the class-wise sum of all relative frequencies is always 100. In addition to
`count` and `%`, `chi_square` is calculated for every feature-class pair. It represents the p-value of
[Pearson's chi-square test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test) when comparing the empirical
distribution of the **non-NA** feature values of each class to their empirical distribution in the remaining classes
("one-vs-rest" schema). Note that the p-values are only inserted into the first row of each feature (corresponding to
the first category of that feature), because they apply to the whole feature rather than individual categories. Also
note that the chi-square test is *not* symmetric between the two distributions to compare.

Consider the following binary classification example with two classes "A" and "B", where statistics of one categorical
feature ("sex") with two categories "female" and "male", and some NA values, are displayed.

| Feature | Value  | A - count | A - % | A - chi_square | B - count | B - % | B - chi_square |
|---------|--------|----------:|------:|---------------:|----------:|------:|---------------:|
| sex     | female |     10048 | 62.33 |       2.05e-19 |       204 | 64.76 |         0.2206 |
|         | male   |      6045 | 37.50 |                |       106 | 33.65 |                |
|         | nan    |        27 |  0.17 |                |         5 |  1.59 |                |