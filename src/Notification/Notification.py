import os 

class __Base__:

    def __init__(self):
        self.VerboseLevel = 3
        self.Caller = ""

    def Format(self, Message, State):
        txt = "\033[0;9" + self._Color + "m"
        Message = self.Verbosity(Message)
        if Message:
            return 
        txt += "" if self.Caller == "" else self.Caller.upper()




    def Verbosity(self, message):
        lvl = len(message) - len(message.lstrip("!"))
        if lvl > self.VerboseLevel:
            return True
        return message.lstrip("!")


class __Failure:

    def __init__(self):
        pass

    
