from .Notification import Notification

class _Condor(Notification):

    def __init__(self):
        pass

    def ProjectInheritance(self, instance):
        if self.Verbose != 0: instance.Verbose = self.Verbose
        if self.ProjectName == None: 
            self.Warning("Inheriting the project name: " + instance.ProjectName)
            self.ProjectName = instance.ProjectName
        if self.ProjectName != instance.ProjectName:
            if instance.ProjectName != "UNTITLED":
                self.Warning("Renaming incoming project name. If this is unintentional, make sure to instantiate a new Condor instance with the other project name.")
            instance.ProjectName = self.ProjectName

        if self.OutputDirectory != None: instance.OutputDirectory = self.OutputDirectory
        elif instance.OutputDirectory != None: self.OutputDirectory = instance.OutputDirectory
        else:
            self.OutputDirectory = "./"
            self.Warning("Variable OutputDirectory undefined, assuming current working directory; " + self.pwd)
    
    def RunningJob(self, name): self.Success("Running job: " + name)
    def DumpedJob(self, name, direc): self.Success("Dumped Job: " + name + " to: " + direc)
