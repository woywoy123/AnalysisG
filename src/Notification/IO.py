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
        
