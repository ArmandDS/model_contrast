[![Test & Upload to TestPyPI](https://github.com/ArmandDS/model_contrast/actions/workflows/test_and_upload_to_TestPyPI.yml/badge.svg)](https://github.com/ArmandDS/model_contrast/actions/workflows/test_and_upload_to_TestPyPI.yml)

[![Upload to PIP](https://github.com/ArmandDS/model_contrast/actions/workflows/upload-to-pip.yml/badge.svg)](https://github.com/ArmandDS/model_contrast/actions/workflows/upload-to-pip.yml)

![License: MIT](https://img.shields.io/github/license/armandds/model_contrast)

# Models Contrast

A simple package for compare the performance of two ML models in sklearn, python.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install model-contrast.

```bash
pip install model-contrast
```

# Usage

## Compare 2 Binary Classifiers

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# create two demo models
X, y =  make_classification(n_samples=700, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
         test_size=0.2, random_state=42)

model1 = RandomForestClassifier(n_estimators=10, random_state=42)
model2 = LogisticRegression()

#train the models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

```

Now let's compare them with our package:

```python
from model_contrast import classificator_contrast

classificator_contrast(model1, model2, X_test, y_test)

```
and it return: 

![image](https://github.com/ArmandDS/model_contrast/blob/main/img/binary.PNG)

**For binary classifiers it computes:**
- The prediction time
- The model's memory allocation
- The Accuracy
- The Roc AUC score
- The F1-Score
- Precision
- Recall
- The Jaccard Score
- The Matthews Correlation
- The Concordance
- The Discordance
- Somer D
- The McNemar's test for compare the classifier

## Compare Multi-Class Classifiers

```python

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# create two demo models
X, y =  make_classification(n_samples=700, random_state=42, n_classes=4, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y,
         test_size=0.2, random_state=42)

model1 = RandomForestClassifier(n_estimators=10, random_state=42)
model2 = LogisticRegression()

#train the models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)


```
Compare them:

```python
from model_contrast import classificator_contrast

classificator_contrast(model1, model2, X_test, y_test)

```
and it returns:

![image](https://github.com/ArmandDS/model_contrast/blob/main/img/multiclass.PNG)

**For multiclass classifiers it computes:**
- The prediction time
- The model's memory allocation
- The Accuracy
- The Roc AUC weighted
- The Roc AUC Macro
- The F1 weighted
- The F1 Macro
- The Precision weighted
- The Precision Macro
- The Recall weighted
- The Recall Macro
- The Jaccard Score
- The Matthews Correlation


## Compare 2 Regressors

```python

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#create the regressor
X, y =  make_regression(n_samples=700, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
         test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=10, random_state=42)
model2 = LinearRegression()

#train the regressors
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)


```
Compare them:

```python
from model_contrast import  regressor_contrast

regressor_contrast(model1, model2, X_test, y_test, threshold=10)

```
and it returns:

![image](https://github.com/ArmandDS/model_contrast/blob/main/img/regressors.PNG)


**For regression models it computes:**
- The prediction time
- The model's memory allocation
- The R2
- The MAE
- The RMSE
- The Explained Variance
- The Shapiro test over residuals
- The D'agostino test over residuals
- The Ratio of prediction between certain value

__The Ratio of prediction between certain value:__ it refers for example, you want to know how many preditions has errors below a threshold (for example below 100, it depends of the problem of course), useful to get a idea about the model behavior.

## To do
Add support for another ML frameworks, not just scikit-learn.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)