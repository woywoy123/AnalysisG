from .Notification import Notification

class FeatureAnalysis(Notification):

    def __init__(self):
        pass

    def ReturnWarning(self, Failed):
        
        self.Warning("------------- Feature Errors -------------------")
        for i in list(set(Failed)):
            self.Warning(i)
    
        self.Warning("------------------------------------------------")
        if len(list(set(Failed))) == int(len(Features)):
            self.Fail("NONE OF THE FEATURES PROVIDED WERE SUCCESSFUL!")
