import time 

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.VerboseLevel = 3
        self.__INFO = "INFO"
        self.__FAIL = "FAILURE"
        self.__WARNING = "WARNING"
        self.Caller = ""
       
        self.__ESC = '\033['

        self.__BOLD = '1'
        self.__NORM = '0'
        self.__CEND = '\033[0m'
        self.__RED = '91'
        self.__GREEN = '92'
        self.__YELLOW = '93'
        self.__it = 0
        self.__i = 0
        self.AllReset = True
        self.NotifyTime = 10
        self.Rate = -1
        self.len = -1
        self.__Color = ""
        self.__FONT = ""
        self.__Text = ""

    def __ColorFormat(self):
        return self.__ESC + self.__FONT + ";" + self.__Color + "m" + self.__Text + self.__CEND


    def Notify(self, Message):
        if len(Message) - len(Message.lstrip("!")) > self.VerboseLevel:
            return 

        Message = Message.lstrip("!")
        self.__Color = self.__GREEN
        if self.Verbose and self.Caller == "":
            self.__Text = self.__INFO
            self.__FONT = "0"
            self.__Text += "::" + Message
            self.__Text = self.__ColorFormat()
        else:
            self.__FONT = self.__BOLD
            self.__Text += self.Caller.upper() + "::" + self.__INFO + "::"
            Text = self.__ColorFormat() 
            self.__Text = Message
            self.__Color = self.__GREEN
            self.__FONT = self.__NORM 
            self.__Text = Text + self.__ColorFormat()
        print(self.__Text)
        self.__Text = ""


    def Warning(self, text):
        self.__Color = self.__YELLOW
        self.__FONT = self.__BOLD
        self.__Text += self.Caller.upper() + "::" + self.__WARNING + "::"
        Text = self.__ColorFormat() 
        self.__Text = text
        self.__Color = self.__YELLOW
        self.__FONT = self.__NORM 
        self.__Text = Text + self.__ColorFormat()
        print(self.__Text)
        self.__Text = ""

    def Fail(self, text):
        self.__Color = self.__RED
        self.__FONT = self.__BOLD
        self.__Text += self.Caller.upper() + "::" + self.__FAIL + "::"
        Text = self.__ColorFormat() 
        self.__Text = text
        self.__Color = self.__RED
        self.__FONT = self.__NORM 
        self.__Text = Text + self.__ColorFormat()
        print(self.__Text)
        self.__Text = ""
        import os
        os._exit(1)


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

