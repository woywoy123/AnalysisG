import time 

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.__INFO = "INFO"
        self.__FAIL = "FAILURE"
        self.__WARNING = "WARNING"
        self.Caller = ""
        self.__CEND = '\033[0m'
        self.__RED = '\033[91m'
        self.__GREEN = '\33[32m'
        self.__it = 0
        self.__i = 0
        self.AllReset = True
        self.NotifyTime = 10
        self.Rate = -1
        self.len = -1

    def Notify(self, Message):
        if self.Verbose and self.Caller == "":
            print(self.__GREEN + self.__INFO + Message + self.__CEND)
        elif self.Verbose and self.Caller != "":
            print(self.__GREEN + self.Caller.upper() + "::" + self.__INFO + "::" + Message + self.__CEND)

    def Warning(self, text):
        print(self.__RED + self.__WARNING + self.Caller + " :: " + text + self.__CEND)

    def CheckAttribute(self, Obj, Attr):
        try: 
            getattr(Obj, Attr)
            return True
        except AttributeError:
            return False

    def ProgressInformation(self, Mode):
        if self.AllReset:
            self.__t_start = time.time()
            self.__it = 0
            self.AllReset = False
            self.Rate = -1
      
        cur = time.time()
        
        if cur - self.__t_start >  self.NotifyTime:
            self.Notify("CURRENT " + Mode + " RATE: " + str(round(float(self.__it) / float(cur - self.__t_start)))[0:4] + " /s - PROGRESS: " + str(round(float(self.__i / self.len)*100, 4)) + "%")
            self.Rate = float(self.__it) / float(cur - self.__t_start)
            self.AllReset = True

        self.__i += 1
        self.__it += 1

    def ResetAll(self):
        self.AllReset = True
        self.__i = 0
        self.__it = 0
        self.len = -1
        self.Rate = -1

class Debugging:

    def __init__(self, Threshold = 100):
        self.__Threshold = Threshold 
        self.__iter = 0
        self.Debug = False
    
    def Count(self):
        self.__iter +=1

    def Stop(self):
        if self.__Threshold <= self.__iter and self.__Threshold != -1:
            return True
        else:
            return False

    def ResetCounter(self):
        self.__iter = 0

