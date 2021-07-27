import numpy

class Alerting:
    def __init__(self, endlen = [], verbose = True, Step = 10, INFO = "INFO::Progress "):
        self.__Verbose = verbose
        self.__Step = Step 
        self.current = 0
       
        if isinstance(endlen, numpy.ndarray):
            self.__Max = len(endlen)
        if isinstance(endlen, list):
            self.__Max = len(endlen)
        if isinstance(endlen, int):
            self.__Max = endlen

        self.__a = 0
        self.__INFO = INFO

    def ProgressAlert(self):

        if self.__Verbose == True:
            per = round((float(self.current) / float(self.__Max))*100)
            
            if per > self.__a:
                print(self.__INFO + str(per) + "%")
                self.__a = self.__a + self.__Step
            self.current += 1
    
    def Notice(self, message):
        if self.__Verbose == True:
            print(self.__INFO + "-> "+message)


class WarningAlert:
    def __init__(self):
        self.__Type = ""

    def MixingTypes(self, current, message = "WARNING::MIXING OBJECT TYPES"):
        try:
            if current in self.ExcludeAlert:
                pass
            elif self.__Type == "":
                self.__Type = current
            else:
                raise AttributeError
        except AttributeError:
            if self.__Type not in current:
                print(message + ":: Original -> " + self.__Type + " | Replaced -> " + current)


class ErrorAlert:
    def __init__(self):
        self.expected = ""
        self.given = ""

    def WrongInputType(self, message):
        
        if isinstance(self.given, self.expected) != True:
            print("ERROR::" + message)
            raise TypeError("Wrong Input")

    def MissingBranch(self, Branch, message =""):
        print("ERROR::"+message)
        raise KeyError("Missing a Branch -> Requested: " + Branch)

class Debugging:
    def __init__(self, events = 0, debug = True):
        self.__Events = events
        self.__Debug = debug 
        self.__iter = 0
        self.DebugKill = False
        
        if self.__Debug == True:
            print("WARNING::DEBUG MODE ENABLED!")

    def BreakCounter(self):
        
        if self.__Debug == False:
            return

        self.__iter += 1
        if self.__iter >= self.__Events:
            self.DebugKill = True

    def ResetCounter(self):
        self.__iter = 0
        self.DebugKill = False
