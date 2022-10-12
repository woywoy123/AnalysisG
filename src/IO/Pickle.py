from AnalysisTopGNN.Tools import Tools
import pickle 

class Pickle(Tools):
    def __init__(self):
        pass

    def PickleObject(self, obj, filename, Dir = "_Pickle"):
        filename = self.AddTrailing(filename, ".pkl")
        direc = self.path(filename)
        direc = self.RemoveTrailing(direc, "/")
        filename = self.filename(filename)
         
        if self.RemoveTrailing(self.pwd(), "/") == direc:
            direc += "/" + Dir
        
        self.mkdir(direc)
        f = open(direc + "/" + filename, "wb")
        pickle.dump(obj, f)
        f.close()
    
    def UnpickleObject(self, filename, Dir = "_Pickle"):
        filename = self.AddTrailing(filename, ".pkl")
        direc = self.path(filename)
        filename = self.filename(filename)
        if self.RemoveTrailing(self.pwd(), "/") == direc:
            direc += "/" + Dir
        f = open(direc + "/" + filename, "rb")
        obj = pickle.load(f)
        f.close()
        return obj

def _UnpickleObject(filename):
    io = Pickle()
    return io.UnpickleObject(filename)
    
def _PickleObject(obj, filename):
    io = Pickle()
    io.PickleObject(obj, filename)
    
