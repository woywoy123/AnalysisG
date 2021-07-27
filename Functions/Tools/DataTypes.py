from multiprocessing import Process, Queue

class DataTypeCheck:

    def __init__(self):
        pass
    
    def AddToList(self, obj):
        output = []
        if isinstance(obj, str):
            output.append(obj)
        if isinstance(obj, list):
            output += obj
        return output

class Threading:
    def __init__(self, lists, threads = 16):
        self.__threads = threads
        self.__lists = lists
        self.Result = []

    def StartWorkers(self):
        
        q = Queue()
        Processes = []
        self.Result = []
        
        sub_p = []
        res = []
        for i in range(len(self.__lists)):
            P = Process(target = self.__lists[i].Runner, args=(q,))
            Processes.append(P)
            sub_p.append(P) 

            P.start()

            if len(sub_p) == self.__threads:
                for p in sub_p:
                    re = q.get()
                    res.append(re)
                sub_p = []
        
        for p in sub_p:
            re = q.get()
            res.append(re)
        
        for i in range(len(res)):
            self.__lists[i].SetResults(res[i])
        
        for p in Processes:
            p.join()
        
        self.Result = self.__lists



class TemplateThreading:
    def __init__(self, name, source_name, target_name, source_value, function):
        self.__name = name 
        self.__source_name = source_name
        self.__target_name = target_name
        self.__source_value = source_value
        self.__function = function

    def Runner(self, q):
        self.__result = self.__function(self.__source_value)
        q.put(self.__result)
    
    def SetResults(self, res):
        self.__result = res

    def SetAttribute(self, obj):
        j = getattr(obj, self.__target_name)
        j[self.__name] = self.__result
        setattr(obj, self.__target_name, j)

