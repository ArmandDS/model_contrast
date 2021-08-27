import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from model_contrast import  regressor_contrast, percentaje_acc

class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.X, self.y =  make_regression(n_samples=700, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
         test_size=0.2, random_state=42)
        self.model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model2 = LinearRegression()
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        self.predict_m1 = self.model1.predict(self.X_test)
        self.predict_m2 = self.model2.predict(self.X_test)
        self.table = regressor_contrast(self.model1, self.model2, self.X_test, self.y_test)

    def test_test_return_df(self):
        self.assertEqual(type(self.table), pd.DataFrame)
        self.assertEqual(self.table.shape, (2, 12))
        self.assertEqual(self.table.columns.tolist(), ["Model", "Exec time(seg)","Memory (Kb)", 
        	"R2","MAE", "RMSE", "Explained Variance", "Residual Shapiro Test Stat", "Residual Shapiro Test p-value",
            "Residual D’Agostino’s Statical", "Residual D’Agostino’s p-value", "Ratio errors in Threshold"])

    def test_sk_metrics(self):
        self.assertEqual(self.table["R2"][0], round(r2_score(self.y_test, self.predict_m1),5))
        self.assertEqual(self.table["MAE"][0], round(mean_absolute_error(self.y_test, self.predict_m1),5))
        self.assertEqual(self.table["RMSE"][0], round(mean_squared_error(self.y_test, self.predict_m1, squared = False),5))
        self.assertEqual(self.table["Explained Variance"][0], round(explained_variance_score(self.y_test, self.predict_m1),5))
        self.assertIsInstance(self.table["Residual Shapiro Test Stat"][0], float)
        self.assertLessEqual(self.table["Residual Shapiro Test p-value"][0], 1)
        self.assertIsInstance(self.table["Residual D’Agostino’s Statical"][0], float)
        self.assertLessEqual(self.table["Residual D’Agostino’s p-value"][0], 1)  
        self.assertEqual(self.table["R2"][1], r2_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["MAE"][1], round(mean_absolute_error(self.y_test, self.predict_m2),5))
        self.assertEqual(self.table["RMSE"][1], round(mean_squared_error(self.y_test, self.predict_m2, squared = False),5))
        self.assertEqual(self.table["Explained Variance"][1], round(explained_variance_score(self.y_test, self.predict_m2),5))
        self.assertIsInstance(self.table["Residual D’Agostino’s Statical"][1], float)
        self.assertLessEqual(self.table["Residual D’Agostino’s p-value"][1], 1) 

    def test_percentaje_acc(self):
        prediction1 = np.array([1000, 1100, 400, 500, 800, 300, 1200, 650, 700, 800])
        label_true = np.array([1000, 1200, 500, 550, 730, 8000, 200, 6500, 900, 300])
        threshold = 100
        self.assertEqual(percentaje_acc(prediction1, label_true, threshold), .5)
        prediction1 = np.array([1000, 1100, 400, 500, 800, 300, 1200, 650, 700, 800])
        label_true = np.array([1000, 1100, 400, 500, 800, 300, 1200, 650, 700, 800])
        threshold = 100
        self.assertEqual(percentaje_acc(prediction1, label_true, threshold), 1)