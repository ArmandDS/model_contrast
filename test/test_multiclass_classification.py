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
from sklearn.preprocessing import label_binarize

from model_contrast import classificator_contrast

class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.X, self.y =  make_classification(n_samples=700, random_state=42, n_classes=4, n_informative=4)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
         test_size=0.2, random_state=42)
        self.model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model2 = LogisticRegression()
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        self.predict_raw_m1 = self.model1.predict(self.X_test)
        self.predict_raw_m2 = self.model2.predict(self.X_test)
        self.table = classificator_contrast(self.model1, self.model2, self.X_test, self.y_test)
        self.predict_m1 = label_binarize(self.predict_raw_m1 , classes=np.unique(self.y_test))
        self.predict_m2 = label_binarize(self.predict_raw_m2 , classes=np.unique(self.y_test))
        self.y_test = label_binarize(self.y_test, classes=np.unique(self.y_test))  

    def test_test_return_df(self):
        self.assertEqual(type(self.table), pd.DataFrame)
        self.assertEqual(self.table.shape, (2, 14))
        self.assertEqual(self.table.columns.tolist(), ["Model", "Exec time(seg)","Memory (Kb)", "Accuracy",
        	"Roc Auc weighted","Roc Auc macro", "F1-Score weighted", "F1-Score macro", "Precision weighted",
            "Precision macro", "Recall weighted","Recall macro","Jaccard", "Matthews_corrcoef"])

    def test_sk_metrics(self):
        self.assertEqual(self.table["Roc Auc weighted"][0], roc_auc_score(self.y_test, self.predict_m1, average='weighted', multi_class = 'ovr'))
        self.assertEqual(self.table["Accuracy"][0], accuracy_score(self.y_test, self.predict_m1))
        self.assertEqual(self.table["F1-Score weighted"][0], f1_score(self.y_test, self.predict_m1, average='weighted'))
        self.assertEqual(self.table["Precision weighted"][0], precision_score(self.y_test, self.predict_m1, average='weighted'))
        self.assertEqual(self.table["Recall weighted"][0], recall_score(self.y_test, self.predict_m1, average='weighted'))
        self.assertEqual(self.table["Jaccard"][0], jaccard_score(self.y_test, self.predict_m1, average= 'macro'))   
        self.assertEqual(self.table["Roc Auc weighted"][1], roc_auc_score(self.y_test, self.predict_m2, average='weighted', multi_class = 'ovr'))
        self.assertEqual(self.table["Accuracy"][1], accuracy_score(self.y_test, self.predict_m2))
        self.assertEqual(self.table["F1-Score weighted"][1], f1_score(self.y_test, self.predict_m2, average='weighted'))
        self.assertEqual(self.table["Precision weighted"][1], precision_score(self.y_test, self.predict_m2, average='weighted'))
        self.assertEqual(self.table["Recall weighted"][1], recall_score(self.y_test, self.predict_m2, average='weighted'))
        self.assertEqual(self.table["Jaccard"][1], jaccard_score(self.y_test, self.predict_m2, average= 'macro'))   
        self.assertEqual(self.table["Roc Auc macro"][0], roc_auc_score(self.y_test, self.predict_m1, average='macro', multi_class = 'ovr'))
        self.assertEqual(self.table["F1-Score macro"][0], f1_score(self.y_test, self.predict_m1, average='macro'))
        self.assertEqual(self.table["Precision macro"][0], precision_score(self.y_test, self.predict_m1, average='macro'))
        self.assertEqual(self.table["Recall macro"][0], recall_score(self.y_test, self.predict_m1, average='macro'))
        self.assertEqual(self.table["Roc Auc macro"][1], roc_auc_score(self.y_test, self.predict_m2, average='macro', multi_class = 'ovr'))
        self.assertEqual(self.table["F1-Score macro"][1], f1_score(self.y_test, self.predict_m2, average='macro'))
        self.assertEqual(self.table["Precision macro"][1], precision_score(self.y_test, self.predict_m2, average='macro'))
        self.assertEqual(self.table["Recall macro"][1], recall_score(self.y_test, self.predict_m2, average='macro'))

