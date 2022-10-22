from sklearn.metrics import roc_curve, auc

class Metrics:
    def __init__(self):
        pass

    def MakeROC(self, feature, truth, pred):
        truth = self.ROC[feature]["truth"]
        truth = torch.cat(truth, dim = 0).view(-1)
        truth = truth.detach().cpu().numpy()
        p_score = self.ROC[feature]["pred_score"]
        p_score = torch.cat([p.softmax(dim = 1).max(1)[0] for p in p_score], dim = 0)
        p_score = p_score.detach().cpu().numpy()
        
        fpr, tpr, _ = roc_curve(truth, p_score)
        auc_ = auc(fpr, tpr)

        self.ROC[feature]["fpr"] += fpr.tolist()
        self.ROC[feature]["tpr"] += tpr.tolist()
        self.ROC[feature]["auc"].append(float(auc_))

        return self.ROC
 
