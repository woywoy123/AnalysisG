import torch

class VariableManager:

    def __init__(self):
        pass

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.__dict__.keys() != other.__dict__.keys():
            return False

        for i, j in zip(self.__dict__.values(), other.__dict__.values()):
            if type(i) != type(j):
                return False

            elif isinstance(i, torch.Tensor):
                if torch.equal(i, j) == False:
                    return False     

            elif isinstance(j, float):
                if i != j:
                    return False
            
            elif isinstance(j, int):
                if i != j:
                    return False

            elif isinstance(i, list):
                if i != j: 
                    return False
            else:
                if list(i) != list(j): 
                    return False
        return True


