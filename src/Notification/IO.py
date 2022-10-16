from .Notification import Notification

class IO(Notification):

    def __init__(self):
        pass

    def FileNotFoundWarning(self, Directory, Name):
        self.Warning("File: " + Name + " not found in " + Directory)

    def EmptyDirectoryWarning(self, Directory):
        self.Warning("No samples found in " + Directory.replace("//", "/") + " skipping.")

    def FoundFiles(self, Files):
        for i in Files:
            self.Success("!!Files Found in Directory: " + i + "\n -> " + "\n -> ".join(Files[i]))

    def PickleOutput(self):
        x = self.abs()

    def DumpingObjectName(self, Name):
        self.Success("!!!Dumping: " + Name + " to " + self.Filename +  self._ext)

    def WrongInputMultiThreading(self, Inpt):
        self.Failure("Wrong input, expected a dictionary but got " + str(type(Inpt)) + " returning.")
    
    def MergingHDF5(self, inpt):
        self.Success("!Merging: " + inpt + " to " + self.Filename)
