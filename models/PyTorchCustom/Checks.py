import torch 

def AssertEquivalence(truth, pred, threshold = 0.001):
    diff = truth - pred
    diff = (diff/truth)*100
    if diff < threshold:
        return True 
    print("-> ", diff, truth, pred)
    return False

def AssertEquivalenceList(truth, pred, threshold = 0.001):
    for i, j in zip(truth, pred):
        if AssertEquivalence(i, j) == False:
            return False
    return True

def AssertEquivalenceRecursive(truth, pred, threshold = 0.001):
    try:
        return AssertEquivalence(float(truth), float(pred), threshold)
    except:
        for i, j in zip(truth, pred):
            if AssertEquivalenceRecursive(i, j, threshold) == False:
                return False 
        return True 


def MakeTensor(inpt, device = "cpu"):
    return torch.tensor([inpt], device = device)


