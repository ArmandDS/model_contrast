[![Test & Upload to TestPyPI](https://github.com/ArmandDS/model_contrast/actions/workflows/test_and_upload_to_TestPyPI.yml/badge.svg)](https://github.com/ArmandDS/model_contrast/actions/workflows/test_and_upload_to_TestPyPI.yml)

![License: MIT](https://img.shields.io/github/license/armandds/model_contrast)

# Models Contrast

A simple package for comparer the performance of two ML models in sklearn, python.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

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


## Compare Multi-Class Classifiers

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
and it returns:

![image](https://github.com/ArmandDS/model_contrast/blob/main/img/multiclass.PNG)


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
and it returns:

![image](https://github.com/ArmandDS/model_contrast/blob/main/img/regressors.PNG)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)