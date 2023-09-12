from .Notification import Notification

class _nTupler(Notification):

    def __init__(self):
        pass
 
    def _KeyNotFound(self, var, inpt, others):
        msg = "Specified key/variable not found (" + var + "::" + inpt + "). "
        msg += "Keys available are: \n->"
        msg += "\n->".join(others)
        self.Warning(msg)
        self.Warning("Skipping for now...")

    def _MissingSelectionName(self, selection):
        msg = "Selection name: " + selection + " not found. Use the following syntax:"
        msg += "\n<var> = nTupler \n <var>.This('SelectionName -> variable', 'tree')"
        self.Warning(msg)
        self.Warning("OPTIONS ARE LISTED BELOW:")
        self.Warning("\n-> ".join([""] + self.ShowSelections))

    def _FoundSelectionName(self, selection):
        msg = "!!Selection name: " +  selection + " found."
        self.Success(msg)
