import uproot
import pickle
import h5py
import numpy as np
import torch
from Functions.IO.Files import Directories, WriteDirectory
from Functions.Tools.DataTypes import Threading, TemplateThreading
from Functions.Tools.Alerting import Notification
from Functions.Tools.Variables import RecallObjectFromString

class File(Notification):
    def __init__(self, directory = None, Verbose = False):
        Notification.__init__(self, Verbose = Verbose) 
        
        if directory != None:
            self.Caller = "FILE +--> " + directory
            self.__Dir = directory
            self.__Reader = uproot.open(self.__Dir)       

        self.ArrayBranches = {}
        self.ArrayLeaves = {}
        self.ArrayTrees = {}
        
        self.ObjectBranches = {}
        self.ObjectLeaves = {}
        self.ObjectTrees = {}

        self.Trees = []
        self.Leaves = []
        self.Branches = []

    def DefineDirectory(self, Directory):
        self.__init__(directory = Directory, Verbose = self.Verbose)
  
    def CheckObject(self, Object, Key):
        if Key == -1:
            return False
        try:
            Object[Key]
            return True
        except uproot.exceptions.KeyInFileError:
            return False

    def CheckKeys(self):
        FoundBranches = set()
        FoundLeaves = set()
        
        for i in self.Trees:
            if self.CheckObject(self.__Reader, i) == False:
                self.Warning("SKIPPED TREE -> " + i)
            else:
                self.ObjectTrees[i] = self.__Reader[i]

        for i, j in self.ObjectTrees.items():
            for k in j.iterkeys():
                Key = k.split("/")
                if Key[-1] in self.Branches:
                    self.ObjectBranches[i + "/" + Key[0]] = self.__Reader[i + "/" + Key[0]]
                    FoundBranches.add(Key[-1])
                    continue
                if Key[-1] in self.Leaves and len(Key) > 1:
                    self.ObjectLeaves[i + "/" + Key[0] + "/" + Key[-1]] = self.__Reader[i + "/" + Key[0] + "/" + Key[-1]]
                    FoundLeaves.add(Key[-1])
                    continue

                if Key[-1] in self.Leaves:
                   self.ObjectLeaves[i + "/" + Key[-1]] = self.__Reader[i + "/" + Key[-1]]
                   FoundLeaves.add(Key[-1])

        for i in self.Branches:
            if i not in FoundBranches:
                self.Warning("SKIPPED BRANCH -> " + i)

        for i in self.Leaves:
            if i not in FoundLeaves:
                self.Warning("SKIPPED LEAF -> " + i)


    def ConvertToArray(self):
        def Convert(obj):
            def Rec(val):
                if isinstance(val, list):
                    return [Rec(i) for i in val]                
                elif isinstance(val, dict):
                    return {j : Rec(val[j]) for j in val}
                elif type(val).__module__ == "numpy":
                    return Rec(getattr(val, "tolist", lambda: val)())
                elif type(val).__name__ == "STLVector":
                    return Rec(val.tolist())
                else:
                    return val
            try:                
                return Rec(obj.array(library = "np"))
            except:
                return []

        
        self.Caller = "CONVERTTOARRAY"
        self.Notify("!!STARTING CONVERSION")
        runners = []
        for i in self.ObjectTrees:  
            if self.CheckAttribute(self.ObjectTrees[i], "array") and i not in self.ArrayTrees:
                th = TemplateThreading(i, "ObjectTrees", "ArrayTrees", self.ObjectTrees[i], Convert)
                runners.append(th)
        
        for i in self.ObjectBranches:
            if self.CheckAttribute(self.ObjectBranches[i], "array") and i not in self.ArrayBranches:
                th = TemplateThreading(i, "ObjectBranches", "ArrayBranches", self.ObjectBranches[i], Convert)
                runners.append(th)
        
        for i in self.ObjectLeaves:
            if self.CheckAttribute(self.ObjectLeaves[i], "array") and i not in self.ArrayLeaves:
                th = TemplateThreading(i, "ObjectLeaves", "ArrayLeaves", self.ObjectLeaves[i], Convert)
                runners.append(th)
        
        T = Threading(runners, self)
        T.Verbose = self.Verbose
        T.VerboseLevel = self.VerboseLevel
        T.StartWorkers()
        del T
        del runners
        del self.ObjectLeaves
        del self.ObjectBranches
        self.Trees = list(self.ObjectTrees)
        del self.ObjectTrees

def PickleObject(obj, filename, Dir = "_Pickle"):
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"
   
    d = WriteDirectory()
    d.MakeDir(Dir)
    if Dir.endswith("/"):
        filename = Dir + filename
    else:
        filename = Dir + "/" + filename

    outfile = open(filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename, Dir = "_Pickle"):
    if Dir.endswith("/"):
        Dir = Dir[0:len(Dir)-1]
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"

    filename = Dir + "/" + filename

    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj

class HDF5(WriteDirectory, Directories):

    def __init__(self, OutDir = None, Name = None, Verbose = True, Chained = True):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        Notification.__init__(self, Verbose)

        self.VerboseLevel = 3
        self.Verbose = Verbose
        self.Caller = "WRITE-TO-HDF5"
        self.__Outdir = OutDir
        self.__Name = Name
        self.__File = None
        
        self.__DirectoryStandard()
        self.MakeDir(self.__Outdir + "/" + self.__Name)
        self.__part = 0
        
        if Chained:
            self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + self.__Name + "_" + str(self.__part) + ".hdf5" 
        else:
            self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + self.__Name + ".hdf5"

    def __DirectoryStandard(self):
        if self.__Outdir == None:
            self.__Outdir = "_Pickle"
        if self.__Name == None:
            self.__Name = "UNTITLED"

        if self.__Outdir.endswith("/"):
            self.__Outdir = self.__Outdir.rstrip("/")
        if self.__Name.endswith("/"):
            self.__Name = self.__Name.rstrip("/")

    def SwitchFile(self):
        tmp = self.VerboseLevel
        self.VerboseLevel = 0
        self.EndFile()
        self.__part += 1
        self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + self.__Name + "_" + str(self.__part) + ".hdf5"
        self.StartFile()
        self.VerboseLevel = tmp

    def ReadChain(self):
        f = self.ListFilesInDir(self.__Outdir + "/" + self.__Name + "/")
        tmp = self.VerboseLevel
        output = {}
        for i in f:
            self.VerboseLevel = 0
            self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + i
            self.StartFile("r")
            output |= self.RebuildObject()
            self.EndFile()
            self.VerboseLevel = tmp
            self.Notify("!!!FINISHED READING -> " + i + " / " + str(len(f)))
        return output

    def StartFile(self, Mode = "w"):
        if Mode == "w":
            self.Notify("!!WRITING -> " + self.__FileDirName)    
        if Mode == "r":
            self.Notify("!!READING -> " + self.__FileDirName)

        self.__File = h5py.File(self.__FileDirName, Mode, track_order = True)
        if "__PointerReferences" in self.__File:
            self.__RefSet = self.__File["__PointerReferences"]
        else:
            self.__RefSet = self.__File.create_dataset("__PointerReferences", (1, ), dtype = h5py.ref_dtype)
    
    def OpenFile(self, SourceDir = None, Name = None):
        self.__OutDir = SourceDir
        self.__Name = Name
        self.__DirectoryStandard()
        self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + self.__Name + ".hdf5"
        self.StartFile(Mode = "r")

    def EndFile(self):
        self.Notify("!!CLOSING FILE -> " + self.__FileDirName)
        self.__File.close()
        self.__File = None
        self.__RefSet = None
   
    def __Store(self, Name, key, val):

        if isinstance(val, dict):
            self.__StoreDict(Name, key, val)

        elif isinstance(val, list):
            if len(val) == 0:
                self.__CreateAttribute(Name, key, val)
            else:
                self.__StoreList(Name, key, val)
        elif "Functions" in str(type(val)):
            self.__CreateAttribute(Name, key, [str(val)])

        elif "torch.Tensor" in str(type(val)):
            self.__CreateAttribute(Name, key, val.numpy())

        elif "torch_geometric" in str(type(val)):
            self.__Store(Name, key, val.to_dict()) 

        elif "torch." in str(type(val)):
            self.__Store(Name, key, "")

        else:
            self.__CreateAttribute(Name, key, val)

    def __CreateDataSet(self, RefName):
        self.__RefSet.attrs[RefName] = self.__File.create_dataset(RefName, data = h5py.Empty(None)).ref
    
    def __CreateAttribute(self, RefName, AttributeName, Data):
        if AttributeName in self.__File[RefName].attrs:
            D = self.__File[RefName].attrs[AttributeName]

            if isinstance(D, np.ndarray):
                D = np.append(D, Data)
            else: 
                D = [D, Data]
            self.__File[RefName].attrs[AttributeName] = D
        else:
            self.__File[RefName].attrs[AttributeName] = Data
        return self.__File[RefName].attrs[AttributeName]

    def __StoreDict(self, Name, AttributeName, Value):
        for key, val in Value.items():
            self.__Store(Name, AttributeName + "/#/" + str(key), val) 
        
        if len(Value) == 0:
            self.__Store(Name, AttributeName + "/#/", "")

    def __StoreList(self, Name, AttributeName, Value):
        if any(isinstance(k, str) or isinstance(k, int) or isinstance(k, float) for k in Value):
            self.__CreateAttribute(Name, AttributeName, Value)
            return  
        for i in Value:
            self.__Store(Name, AttributeName, i)
        if len(Value) == 0:
            self.__Store(Name, AttributeName, [])

    def DumpObject(self, obj, Name = None):
        
        if Name == None:
            Name = hex(id(obj))
        
        if Name in self.__RefSet.attrs:
            return 

        type_ = str(type(obj).__module__) + "." + str(type(obj).__name__)
        self.__CreateDataSet(Name)
        self.__CreateAttribute(Name, "__FunctionType", type_)
        for i, j in obj.__dict__.items():
            self.__Store(Name, i, j)

    def RebuildObject(self):
        def Dictify(lst, dic, va):
            if len(lst) == 0:
                return va
             
            l = lst[0] 
            if l.isdigit():
                l = int(l)
           
            if l == "":
                dic |= {}
            elif l not in dic:
                dic |= {l: Dictify(lst[1:], {}, va)}
            else:
                dic[l] |= Dictify(lst[1:], dic[l], va)
            return dic

        obj = self.__File
        output = {}
        
        for i in obj.keys():
            if i == "__PointerReferences":
                continue
                
            obj_type = obj[i].attrs["__FunctionType"]
            target_obj = RecallObjectFromString(obj_type)
            
            out = {}
            attr_List = []
            for key in obj[i].attrs.keys():
                if key == "__FunctionType":
                    continue
                val = obj[i].attrs[key]
                val = getattr(val, "tolist", lambda: val)()
                if "/#/" in key:
                    dic_v = key.split("/#/")
                    key = key.split("/#/")[0]
                    out = Dictify(dic_v, out, val)
                else:
                    setattr(target_obj, key, val)

                if key not in attr_List:
                    attr_List.append(key)
            
            for key in out:
                if type(target_obj).__name__ == "Data":
                    p = {k : torch.tensor(out[key][k]) for k in out[key]}
                    target_obj = target_obj.from_dict(p)
                    attr_List = list(target_obj.__dict__)
                else:
                    setattr(target_obj, key, out[key])
           
            t_list = list(target_obj.__dict__.keys())
            for k in t_list:
                if k not in attr_List:
                    delattr(target_obj, k) 
            output[i] = target_obj
        return output 
