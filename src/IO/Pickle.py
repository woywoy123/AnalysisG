from AnalysisG.Tools import Tools, Threading
import pickle
import sys


class Pickle(Tools):
    def __init__(self):
        self.Caller = "PICKLER"

    def PickleObject(self, obj, filename, Dir="_Pickle"):
        if not filename.endswith(self._ext):
            filename += self._ext

        direc = self.path(filename)
        if direc.endswith("/"): direc = direc[:-1]
        if self.pwd()[:-1] == direc: direc += "/" + Dir
        filename = self.filename(filename)

        self.mkdir(direc)
        f = open(direc + "/" + filename, "wb")
        pickle.dump(obj, f)
        f.close()

    def UnpickleObject(self, filename, Dir="_Pickle"):
        if not filename.endswith(self._ext):
            filename += self._ext
        direc = self.path(filename)
        filename = self.filename(filename)

        if self.pwd()[:-1] == direc: direc += "/" + Dir
        if not self.IsFile(direc + "/" + filename): return

        f = open(direc + "/" + filename, "rb")
        obj = pickle.load(f)
        f.close()
        return obj

def _UnpickleObject(filename):
    return Pickle().UnpickleObject(filename)

def _PickleObject(obj, filename):
    Pickle().PickleObject(obj, filename)
