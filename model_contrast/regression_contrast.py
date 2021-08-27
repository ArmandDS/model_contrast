from time import time
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import shapiro
from scipy.stats import normaltest

def _multiples_normality_test(predictions:np.ndarray, label_data:pd.Series)-> tuple:
  """returns the result from the residual normality Test shapiro and d'agostino"""
  data = label_data - predictions
  shap_sta, shap_p = shapiro(data)
  dagostino_sta, dagostino_p =  normaltest(data)


  return shap_sta, shap_p, dagostino_sta, dagostino_p


def percentaje_acc(predictions:np.ndarray, label_data:pd.Series, threshold:int)-> float:
  """Compute the 'accuracy', i.e given a threshold it return how many predictions error fall below"""
  residuals = abs(label_data - predictions)
  return sum([1 if res<=threshold else 0 for res in residuals])/len(residuals)



def regressor_contrast(model1:RegressorMixin, 
                      model2:RegressorMixin, 
                      test_data:pd.DataFrame, 
                      label_data:pd.Series,
                      threshold:int=10)->pd.DataFrame:
  """Compute 11 metrics to compare a Sckit-learn regression models
  and make statistical test for residual normality"""
  np.random.seed(33)
  models_time = []
  models_memory = []
  models_predictions = []
  models_acc = []
  models_dagostino = []
  models_dagostino_p = []
  models_explained_variance = []
  models_r2 = []
  models_rmse = []
  models_mae = []
  models_shapiro = []
  models_shapiro_p = []


  for m in [model1, model2]:
      t1 = time()
      predictions = m.predict(test_data)
      t2 = time()
      models_time.append(t2 -t1)
      models_predictions.append(predictions)
      models_explained_variance.append(round(explained_variance_score(label_data,predictions),5))
      models_r2.append(round(r2_score(label_data,predictions),5))
      models_rmse.append(round(mean_squared_error(label_data,predictions, squared = False ),5))
      models_mae.append(round(mean_absolute_error(label_data,predictions),5))
      models_acc.append(round(percentaje_acc(label_data,predictions, threshold=threshold),5))

      models_memory.append(sys.getsizeof(m)/1024)
      shap_sta, shap_p, dagostino_sta, dagostino_p = _multiples_normality_test(predictions, label_data)
      models_shapiro.append(round(shap_sta,5))
      models_dagostino.append(round(dagostino_sta,5))
      models_shapiro_p.append(shap_p)
      models_dagostino_p.append(dagostino_p)


  table = pd.DataFrame({
      "Model": ["Model1", "Model2"],
      "Exec time(seg)": models_time,
      "Memory (Kb)": models_memory,
      "R2":models_r2,
      "MAE": models_mae,      
      "RMSE": models_rmse,
      "Explained Variance": models_explained_variance,
      "Residual Shapiro Test Stat": models_shapiro ,
      "Residual Shapiro Test p-value": models_shapiro_p, 
      "Residual D’Agostino’s Statical": models_dagostino ,
      "Residual D’Agostino’s p-value": models_dagostino_p, 
      "Ratio errors in Threshold": models_acc


      })

  return table