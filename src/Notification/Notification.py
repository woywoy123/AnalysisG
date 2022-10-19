import os 

class Base:

    def __init__(self):
        self.VerboseLevel = 3
        self.Caller = ""

    def Format(self, Color, Message, State):

        Message = self.Verbosity(Message)
        if isinstance(Message, bool):
            return 
 
        txt = "\033[0;9" if self.Caller == "" else "\033[1;9"
        txt += str(Color) + "m"
        txt += "" if self.Caller == "" else self.Caller.upper() + "::"
        txt += State + "::"
        txt += Message if self.Caller == "" else "\033[0;9" + str(Color) + "m" + Message 
        txt += "\033[0m"
        print(txt)

    def Verbosity(self, message):
        lvl = len(message) - len(message.lstrip("!"))
        if lvl > self.VerboseLevel:
            return True
        return message.lstrip("!")


class __Failure(Base):

    def __init__(self):
        self.Base.__init__(self)
    
    def Failure(self, Message):
        self.Format(1, Message, "FAILURE")
    
    def FailureExit(self, Message):
        self.Format(1, Message, "FAILURE")
        self.Format(1, "="*len(Message), "FAILURE")
        os._exit(1) 

class __Success(Base):

    def __init__(self):
        self.Base.__init__(self)

    def Success(self, Message):
        self.Format(2, Message, "SUCCESS")

    def FinishExit(self, Message = ""):
        self.Format(2, Message, "FINISHED")
        os._exit(1)
 
class __Warning(Base):

    def __init__(self):
        self.Base.__init__(self)

    def Warning(self, Message):
        self.Format(3, Message, "WARNING")

class Notification(__Failure, __Success, __Warning):

    def __init__(self):
        self.VerboseLevel = 3
        self.Caller = ""
