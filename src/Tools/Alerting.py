import os

class Notification:
    def __init__(self):
        self.VerboseLevel = 3
        self.Caller = ""
    
    def __Format(self, Color, Message, Field):
        if self.VerboseLevel > 0 and self.Caller == "":
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


class OptimizerNotifier:

    def __init__(self):
        pass

    def LongestOutput(self):
        self._len = 0
        for key in self.ModelOutputs:
            if self._len < len(key):
                self._len = len(key)

    def ModelDebug(self, truth_pred, model_pred, Loss, key):

        if self.Debug == False:
            return False
        
        elif self.Debug == True:
            print("-" *10 + "(" + key + ")" +"-"*10)
            print("---> Truth: \n", truth_pred.tolist())
            print("---> Model: \n", model_pred.tolist())
            print("---> DIFF: \n", (truth_pred - model_pred).tolist())
            print("(Loss)---> ", float(Loss))
            return False
        elif self.Debug == "Loss":
            dif = key + " "*int(longest - len(key))
            print(dif + " | (Loss)---> ", float(Loss))
            return False
        elif self.Debug == "Pred":
            print("---> Truth: \n", t_p.tolist())
            print("---> Model: \n", m_p.tolist())
            return False
        else:
            return self.Debug

    
