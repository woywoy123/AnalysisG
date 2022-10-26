from sklearn.metrics import roc_curve, auc
import numpy as np
import statistics 

class Metrics:
    def __init__(self):
        pass

    def MakeROC(self, truth, p_score):
        fpr, tpr, _ = roc_curve(np.array(truth), np.array(p_score))
        auc_ = auc(fpr, tpr)
        
        ROC = {}
        ROC["fpr"] = fpr.tolist()
        ROC["tpr"] = tpr.tolist()
        ROC["auc"] = float(auc_)

        return ROC
    
    def MakeStatics(self, Data):
        if isinstance(Data, dict):
            for i in Data:
                Data[i] = self.MakeStatics(Data[i])
            return Data
        mean = statistics.mean(Data)
        std = statistics.stdev(Data)
        return [mean, std]


