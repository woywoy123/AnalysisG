from torch_geometric.utils import accuracy


class EvaluationMetrics:

    def __init__(self):
        pass
    
    def Accuracy(self, prediction, truth):
        print("Accuracy: ", accuracy(prediction, truth))
