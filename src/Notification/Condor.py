from .Notification import Notification

class Condor(Notification):

    def __init__(self):
        pass

    def ProjectInheritance(self, instance):
        if self.VerboseLevel != 0:
            instance.VerboseLevel = self.VerboseLevel

        if self.ProjectName == None:
            self.Warning("Inheriting the project name: " + instance.ProjectName)
        
        if self.ProjectName != instance.ProjectName:
            if instance.ProjectName != "UNTITLED":
                self.Warning("Renaming incoming project name. If this is unintentional, make sure to instantiate a new Condor instance with the other project name.")
            instance.ProjectName = self.ProjectName

        if self.OutputDirectory != None:
            instance.OutputDirectory = self.OutputDirectory
        else:
            self.OutputDirectory = "./"
            self.Warning("Variable OutputDirectory undefined, assuming current working directory; " + self.pwd())
        
        if self.Tree != None:
            instance.Tree = self.Tree


    def RunningJob(self, name):
        self.Success("Running job: " + name)

    def DumpedJob(self, name, direc):
        self.Success("Dumped Job: " + name + " to: " + direc)
