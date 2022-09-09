import os

class Notification():
    def __init__(self, Verbose = False):
        self.Verbose = Verbose
        self.VerboseLevel = 3
        self.Caller = ""
    
    def __Format(self, Color, Message, Field):
        if self.Verbose and self.Caller == "":
            txt = "\033[0;9" + Color + "m" + Field + "::"
            txt += Message
            txt += "\033[0m" 
        else:
            txt = "\033[1;9" + Color + "m"
            txt += self.Caller.upper() + "::" + Field + "::"
            txt += "\033[0m" 
            txt += "\033[0;9" + Color + "m" + Message + "\033[0m"
        print(txt.replace("//", "/"))

    def Notify(self, Message):
        if len(Message) - len(Message.lstrip("!")) > self.VerboseLevel:
            return 
        Message = Message.lstrip("!")
        self.__Format(str(2), Message, "INFO")

    def Warning(self, Message):
        self.__Format(str(3), Message, "WARNING") 

    def Fail(self, Message):
        self.__Format(str(1), Message, "FAILURE")
        os._exit(1)


