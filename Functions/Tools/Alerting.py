import uproot

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.__INFO = "INFO::"
        self.__FAIL = "FAILURE::"
        self.__WARNING = "WARNING::"
        self.Caller = ""
        self.Alerted = []

    def Notify(self, Message):
        if self.Verbose and self.Caller == "":
            print(self.__INFO + Message)
        elif self.Verbose and self.Caller != "":
            print(self.Caller.upper()+"::"+self.__INFO+"::"+Message)

    def NothingGiven(self, Obj, Name):
        if len(Obj) == 0:
            print(self.Caller.upper()+"::"+self.__WARNING+"::NOTHING GIVEN +-> " + Name)


    def CheckObject(self, Object, Key):
        try:
            Object[Key]
            return True
        except uproot.exceptions.KeyInFileError:
            if Key not in self.Alerted:
                print(self.__WARNING + self.Caller + " :: Key -> " + Key + " NOT FOUND")
                self.Alerted.append(Key)
            return False

    def CheckAttribute(self, Obj, Attr):
        try: 
            getattr(Obj, Attr)
            return True
        except AttributeError:
            return False

class Debugging:

    def __init__(self, Threshold = 100):
        self.__Threshold = Threshold 
        self.iter = 0
    
    def TimeOut(self):
        if self.__Threshold == self.iter:
            exit()
        self.iter +=1
