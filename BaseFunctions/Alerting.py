class Alerting:
    def __init__(self, endlen, verbose = True, Step = 10, INFO = "INFO::Process "):
        self.__Verbose = verbose
        self.__Step = Step 
        self.current = 0
        
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

