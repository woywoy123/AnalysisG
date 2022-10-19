from .Notification import Notification

class Condor(Notification):

    def __init__(self):
        pass

    def ProjectInheritance(self, instance):
        if self.ProjectName == None:
            self.Warning("Inheriting the project name: " + instance.ProjectName)
        else:
            instance.ProjectName = self.ProjectName

    def RunningJob(self, name):
        self.Success("Running job: " + name)

