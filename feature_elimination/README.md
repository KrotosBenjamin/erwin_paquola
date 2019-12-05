feature~elimination~ - A package for preforming feature elimination
===================================================================

This is a package for feature elimination currently only supports random
forest classification.

-   Future implementation will include:
    -   random forest regression
    -   linear regression

Authors: Apuã Paquola, Kynon Benjamin, and Tarun Katipalli

If using please cite: XXX.

Installation
------------

`pip install --user feature_elimination`

Reference Manual
----------------

  Function Name                        Description
  ------------------------------------ ----------------------------------------------------------------------------------------------------
  [feature~elimination~](refsec:one)   Runs random forest classification feature elimination
  [features~rankfnc~](refsec:two)      Rank features
  [n~featuresiter~](refsec:three)      Determines the features to keep
  [oob~predictions~](refsec:four)      Extracts out-of-bag (OOB) predictions from random forest classifier classes
  [oob~scoreaccuracy~](refsec:five)    Calculates the accuracy score for the OOB predictions
  [oob~scorenmi~](refsec:six)          Calculates the normalized mutual information score for the OOB predictions
  [oob~scoreroc~](refsec:seven)        Calculates the area under the ROC curve (AUC) for the OOB predictions
  [plot~acc~](refsec:eight)            Plot feature elimination with accuracy as measurement
  [plot~nmi~](refsec:nine)             Plot feature elimination with NMI as measurement
  [plot~roc~](refsec:ten)              Plot feature elimination with AUC ROC curve as measurement
  [rf~fe~](refsec:eleven)              Iterate over features to be eliminated
  [rf~festep~](refsec:twelve)          Apply random forest to training data, rank features, and conduct feature elimination (single step)
                                       

### feature~elimination~

[]{#refsec:one}

Runs random forest feature elimination step over iterator process.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   features: a vector of feature names
-   fold: current fold
-   out~dir~: output directory. default \'.\'
-   elimination~rate~: percent rate to reduce feature list. default .2
-   RANK: Output feature ranking. default=True (Boolean)

**Yields:**

-   dict: a dictionary with number of features, normalized mutual
    information score, accuracy score, and array of the indexes for
    features to keep

### feature~rankfnc~

[]{#refsec:two}

Ranks features.

**Args:**

-   features: A vector of feature names
-   rank: A vector with feature ranks based on absolute value of feature
    importance
-   n~featurestokeep~: Number of features to keep. (Int)
-   fold: Fold to analyzed. (Int)
-   out~dir~: Output directory for text file. Default \'.\'
-   RANK: Boolean (True or False)

**Yields:**

-   Text file: Ranked features by fold tab-delimited text file, only if
    RANK=True

### n~featuresiter~

[]{#refsec:three}

Determines the features to keep.

**Args:**

-   nf: current number of features
-   keep~rate~: percentage of features to keep

**Yields:**

-   int: number of features to keep

### oob~predictions~

[]{#refsec:four}

Extracts out-of-bag (OOB) predictions from random forest classifier
classes.

**Args:**

-   estimator: Random forest classifier object

**Yields:**

-   vector: OOB predicted labels

### oob~scoreaccuracy~

[]{#refsec:five}

Calculates the accuracy score from the OOB predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: accuracy score

### oob~scorenmi~

[]{#refsec:six}

Calculates the normalized mutual information score from the OOB
predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: normalized mutual information score

### oob~scoreroc~

[]{#refsec:seven}

Calculates the area under the ROC curve score for the OOB predictions.

**Args:**

-   estimator: Random forest classifier object
-   Y: a vector of sample labels from training data set

**Yields:**

-   float: AUC ROC score

### plot~acc~

[]{#refsec:eight}

Plot feature elimination results for accuracy.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out~dir~: output directory. default \'.\'

**Yields:**

-   graph: plot of feature by accuracy, automatically saves files as png
    and svg

### plot~nmi~

[]{#refsec:nine}

Plot feature elimination results for normalized mutual information.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out~dir~: output directory. default \'.\'

**Yields:**

-   graph: plot of feature by NMI, automatically saves files as png and
    svg

### plot~roc~

[]{#refsec:ten}

Plot feature elimination results for AUC ROC curve.

**Args:**

-   d: feature elimination class dictionary
-   fold: current fold
-   out~dir~: output directory. default \'.\'

**Yields:**

-   graph: plot of feature by AUC, automatically saves files as png and
    svg

### rf~fe~

[]{#refsec:eleven}

Iterate over features to by eliminated by step.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n~featuresiter~: iterator for number of features to keep loop
-   features: a vector of feature names
-   fold: current fold
-   out~dir~: output directory. default \'.\'
-   RANK: Boolean (True or False)

**Yields:**

-   list: a list with number of features, normalized mutual information
    score, accuracy score, and array of the indices for features to keep

### rf~festep~

[]{#refsec:twelve}

Apply random forest to training data, rank features, conduct feature
elimination.

**Args:**

-   estimator: Random forest classifier object
-   X: a data frame of training data
-   Y: a vector of sample labels from training data set
-   n~featurestokeep~: number of features to keep
-   features: a vector of feature names
-   fold: current fold
-   out~dir~: output directory. default \'.\'
-   RANK: Boolean (True or False)

**Yields:**

-   dict: a dictionary with number of features, normalized mutual
    information score, accuracy score, and selected features
