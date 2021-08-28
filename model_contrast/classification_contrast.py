from time import time
import sys
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar

def calculate_concor_discor(predictions_proba:np.ndarray, 
                                label_data:pd.Series)-> tuple:
    """returns the concordance, the discordance and the somers"""

    Probability_df = pd.DataFrame(predictions_proba)
    Probability_df.columns = ['Prob_0','Prob_1']
    truth_table = pd.DataFrame({"real": label_data, 
                        "pred": Probability_df['Prob_1']})
    zeros = truth_table[(truth_table['real']==0)].reset_index().drop(['index'], axis = 1)
    ones = truth_table[(truth_table['real']==1)].reset_index().drop(['index'], axis = 1)
    zeros_list = sorted([zeros.iloc[j,1] for j in zeros.index])
    zeros_length = len(zeros_list)
    zeros_length 
    disc = 0
    ties = 0
    conc = 0
    for i in ones.index:
        cur_conc = bisect_left(zeros_list, ones.iloc[i,1])
        cur_ties = bisect_right(zeros_list, ones.iloc[i,1]) - cur_conc
        conc += cur_conc
        ties += cur_ties
    pairs_tested = zeros_length * len(ones.index)
    disc = pairs_tested - conc - ties
    concordance = conc/pairs_tested
    discordance = disc/pairs_tested
    Somers= concordance - discordance

    return concordance, discordance, Somers


def mcnemar_test(prediction1:np.ndarray,
                    prediction2:np.ndarray, 
                    label_data:pd.Series )-> tuple:
    """compute the Mcnemar's test"""    

    df_c = pd.DataFrame({"A":prediction1, "B": prediction2 })
    ct = pd.crosstab(df_c.A == label_data, df_c.B == label_data)
    table = [[ct[1][1], ct[0][1]],
                [ct[1][0], ct[0][0]]]

    result = mcnemar(table, exact=True)

    return result.pvalue, result.statistic


def _comparator(model1:ClassifierMixin, 
                      model2:ClassifierMixin, 
                      test_data:pd.Series, 
                      label_data:pd.Series,
                      isbinary:bool)->pd.DataFrame:

    """Compute 13 metrics to compare a Sckit-learn clasification models
    and make Statistical test looking for if there is difference in the predictions """
    np.random.seed(33)
    models_time = []
    models_predictions = []

    if isbinary:
        table = pd.DataFrame(columns=["Model", "Exec time(seg)","Memory (Kb)", "Accuracy",
        	"Roc Auc","F1-Score", "Precision", "Recall", "Jaccard", "Matthews_corrcoef",
            "Concordance", "Discordance", "Somer D"])
    else:
        table = pd.DataFrame(columns=["Model", "Exec time(seg)","Memory (Kb)", "Accuracy",
        	"Roc Auc weighted","Roc Auc macro", "F1-Score weighted", "F1-Score macro", "Precision weighted",
            "Precision macro", "Recall weighted","Recall macro","Jaccard", "Matthews_corrcoef"])

    table.loc[len(table)] =  [""]*len(table.columns)
    table.loc[len(table)] =  [""]*len(table.columns)
    for i,m in enumerate([model1, model2]):
        t1 = time()
        proba_pred = m.predict_proba(test_data)
        t2 = time()
        models_time.append(t2 -t1)
        predictions = np.argmax(proba_pred, axis=1)
        models_predictions.append(predictions)
       
        if isbinary:
            concordace, discordance, Somer = calculate_concor_discor(proba_pred, label_data)
            table.iloc[i]['Concordance'] = concordace
            table.iloc[i]['Discordance'] = discordance
            table.iloc[i]['Somer D'] = Somer
            table.iloc[i]['Roc Auc'] = roc_auc_score(label_data,predictions)
            table.iloc[i]['F1-Score'] = f1_score(label_data,predictions)
            table.iloc[i]['Precision'] = precision_score(label_data,predictions)
            table.iloc[i]['Recall'] = recall_score(label_data,predictions)
            table.iloc[i]['Jaccard'] = jaccard_score(label_data,predictions)

        else:
            ytest = label_binarize(label_data, classes=np.unique(label_data))
            predictions_label = label_binarize(predictions, classes=np.unique(label_data))           
            table.iloc[i]['Roc Auc weighted'] = roc_auc_score(ytest,predictions_label, average= 'weighted', multi_class = 'ovr')
            table.iloc[i]['F1-Score weighted'] = f1_score(ytest,predictions_label, average= 'weighted')
            table.iloc[i]['Precision weighted'] = precision_score(ytest,predictions_label, average= 'weighted')
            table.iloc[i]['Recall weighted'] = recall_score(ytest,predictions_label, average= 'weighted')
            table.iloc[i]['Roc Auc macro'] = roc_auc_score(ytest,predictions_label, average= 'macro', multi_class = 'ovr')
            table.iloc[i]['F1-Score macro'] = f1_score(ytest,predictions_label, average= 'macro')
            table.iloc[i]['Precision macro'] = precision_score(ytest,predictions_label, average= 'macro')
            table.iloc[i]['Recall macro'] = recall_score(ytest,predictions_label, average= 'macro')
            table.iloc[i]['Jaccard'] = jaccard_score(ytest,predictions_label, average= 'macro')
        
        table.iloc[i]['Exec time(seg)'] = t2 -t1
        table.iloc[i]['Accuracy'] = accuracy_score(label_data,predictions)
        table.iloc[i]['Matthews_corrcoef'] = matthews_corrcoef(label_data,predictions)
        
        table.iloc[i]['Memory (Kb)'] = sys.getsizeof(m)/1024

    if isbinary:
        pvalue, stac = mcnemar_test(models_predictions[0], models_predictions[1], label_data)
        mn_row = [""]*len(table.columns)
        mn_row1 = [""]*len(table.columns)
        mn_row2 = [""]*len(table.columns)
        mn_row[0] = "The McNemar's Test:"
        mn_row1[0]  = "p-value"
        mn_row1[1]  = pvalue
        mn_row2[0] = "statistic"
        mn_row2[1] = stac
        table.loc[len(table)] =mn_row
        table.loc[len(table)] =mn_row1
        table.loc[len(table)] =mn_row2
    return table




def classificator_contrast(model1:ClassifierMixin, 
                      model2:ClassifierMixin, 
                      test_data:pd.Series, 
                      label_data:pd.Series)->pd.DataFrame:
    """Main methods for compute the metrics """

    isbinary = len(np.unique(label_data)) == 2 
    table =  _comparator(model1, model2, test_data, label_data, isbinary)

    return table               



