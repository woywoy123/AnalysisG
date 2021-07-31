
class VariableManager:

    def __init__(self):
        pass

    def ListAttributes(self):
        self.Branches = []
        for i in list(self.__dict__.values()):
            if isinstance(i, str) and i != "" and i != self.Type:
                self.Branches.append(i)

    def SetAttribute(self, key, val):
        setattr(self, key, val)


