import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score


from model_contrast import classificator_contrast, calculate_concor_discor, mcnemar_test

class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.X, self.y =  make_classification(n_samples=700, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
         test_size=0.2, random_state=42)
        self.model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model2 = LogisticRegression()
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        self.predict_m1 = self.model1.predict(self.X_test)
        self.predict_m2 = self.model2.predict(self.X_test)
        self.table = classificator_contrast(self.model1, self.model2, self.X_test, self.y_test)

    def test_test_return_df(self):
        self.assertEqual(type(self.table), pd.DataFrame)
        self.assertEqual(self.table.shape, (5, 13))
        self.assertEqual(self.table.columns.tolist(), ["Model", "Exec time(seg)","Memory (Kb)", "Accuracy",
        	"Roc Auc","F1-Score", "Precision", "Recall", "Jaccard", "Matthews_corrcoef",
            "Concordance", "Discordance", "Somer D"])

    def test_sk_metrics(self):
        self.assertEqual(self.table["Roc Auc"][0], roc_auc_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["Accuracy"][0], accuracy_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["F1-Score"][0], f1_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["Precision"][0], precision_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["Recall"][0], recall_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["Jaccard"][0], jaccard_score(self.y_test, self.predict_m1))   
        self.assertEqual(self.table["Roc Auc"][1], roc_auc_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["Accuracy"][1], accuracy_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["F1-Score"][1], f1_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["Precision"][1], precision_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["Recall"][1], recall_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["Jaccard"][1], jaccard_score(self.y_test, self.predict_m2))   


    def test_metric_concordance_discordance(self):
        self.assertEqual( calculate_concor_discor(np.array([[0.3, 0.70], [0.27, 0.63], [0.77,0.23],[0.85, 0.15]]), np.array([1,0,1,0])), (.75, .25, .5))

    def test_mcnemar_test(self):
        prediction1 = np.array([1,0,1,1,1,1,1,1,0,0])
        prediction2 = np.array([0,0,0,1,1,1,1,1,1,0])
        label_true = pd.Series([1,1,0,0,1,1,1,0,0,0])
        self.assertEqual( mcnemar_test(prediction1, prediction2, label_true), (1.0, 1.0))
