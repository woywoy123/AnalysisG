import uproot
import pickle
import h5py
import numpy as np
import torch
from AnalysisTopGNN.IO  import Directories, WriteDirectory
from AnalysisTopGNN.Tools import Threading, TemplateThreading
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.Tools import RecallObjectFromString


class File(Notification):
    def __init__(self, FileDirectory):
        Notification.__init__(self)
        self.Caller = "FILE"
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self._Reader =  uproot.open(FileDirectory)
        self._State = None
        self._Threads = 1
   
    def _CheckKeys(self, List, Type):
        TMP = []

        t = []
        for i in [k for k in self._State.keys() if "/" not in k]:
            if ";" in i:
                i = i.split(";")[0]
            t.append(i)

        for i in List:
            if i not in t:
                continue
            TMP.append(i)
            
        return TMP
    
    def _GetKeys(self, List1, List2, Type):
        TMP = []
        for i in List1:
            self._State = self._Reader[i]
            TMP += [i + "/" + j for j in self._CheckKeys(List2, Type)]
        
        for i in TMP:
            if i.split("/")[-1] in List2:
                List2.pop(List2.index(i.split("/")[-1]))
        
        if len(List1) == 0:
            return []
        for i in List2:
            self.Warning("SKIPPED " + Type + ": " + i)
        
        return TMP

    def _GetBranches(self):
        return self._GetKeys(self.Trees, self.Branches, "BRANCH")
    
    def _GetLeaves(self):
        leaves = []
        leaves += self._GetKeys(self.Branches, self.Leaves, "LEAF")
        leaves += self._GetKeys(self.Trees, self.Leaves, "LEAF")
        return leaves 
    
    def ValidateKeys(self):
        self.Leaves = list(set(self.Leaves))
        self.Branches = list(set(self.Branches))
        self.Trees = list(set(self.Trees))
        self._State = self._Reader 
        self.Trees = self._CheckKeys(self.Trees, "TREE")
        self.Branches = self._GetBranches()
        self.Leaves = self._GetLeaves()     
    
    def GetTreeValues(self, Tree):
        All = []
        All += [b.split("/")[-1] for b in self.Branches]
        All += [l.split("/")[-1] for l in self.Leaves]
        for i in self._Reader[Tree].iterate(All, library = "ak"):
            self.Iter = i.to_list()
            break

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

    if filename.endswith(".pkl") and Dir == "_Pickle" and len(filename.split("/")) > 1:
        l = filename.split("/")
        Dir = "/".join(l[:-1])
        filename = l[-1]
    elif filename.endswith(".pkl"):
        filename = filename.split("/")[-1]
    else:
        filename += ".pkl"

    filename = Dir + "/" + filename
    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj

class HDF5(WriteDirectory, Directories):

    def __init__(self, OutDir = "_Pickle", Name = "UNTITLED", Verbose = True, Chained = False):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        Notification.__init__(self, Verbose)

        self.VerboseLevel = 3
        self.Verbose = Verbose
        self.Caller = "WRITE-TO-HDF5"
        
        self._OutDir = OutDir
        self._Name = Name
        self._Chained = Chained
        self._part = 0

    def __DirectoryStandard(self):
        if self._OutDir.endswith("/"):
            self._OutDir = self._OutDir[:-1]
        if self._Name.endswith("/"):
            self._Name = self._Name[:-1]
        
        if self._Chained == True:
            self._FileDirName = self._OutDir + "/" + self._Name + "/Part_" + str(self._part)
            self.MakeDir(self._OutDir + "/" + self._Name)
        else:
            self._FileDirName = self._OutDir + "/" + self._Name
            self.MakeDir(self._OutDir)

        self._FileDirName += ".hdf5"

    def SwitchFile(self):
        tmp = self.VerboseLevel
        self.VerboseLevel = 0
        self.EndFile()
        self._part += 1
        self.__DirectoryStandard()
        self.StartFile()
        self.VerboseLevel = tmp

    def __ReadChain(self):
        self.__DirectoryStandard()
        f = self.ListFilesInDir(self._OutDir + "/" + self._Name + "/")
        tmp = self.VerboseLevel
        output = {}
        for self._FileDirName in f:
            self.VerboseLevel = 0
            self.StartFile("r")
            output |= self.RebuildObject()
            self.EndFile()
            self._part += 1
            self.VerboseLevel = tmp
            name = int(self._FileDirName.replace(".hdf5", "").split("_")[-1])+1
            self.Notify("!!!FINISHED READING -> " + str(name) + " / " + str(len(f)))
        return output

    def StartFile(self, Mode = "w"):
        
        self.__DirectoryStandard()
        if Mode == "w":
            self.Notify("!!WRITING -> " + self._FileDirName)    
        if Mode == "r":
            self.Notify("!!READING -> " + self._FileDirName)
        
        self._File = h5py.File(self._FileDirName, Mode, track_order = True)
        if "__PointerReferences" in self._File:
            self._RefSet = self._File["__PointerReferences"]
        else:
            self._RefSet = self._File.create_dataset("__PointerReferences", (1, ), dtype = h5py.ref_dtype)
    
    def OpenFile(self, SourceDir = "_Pickle", Name = "UNTITLED"):
        self._OutDir = SourceDir
        self._Name = Name

        if self._Chained == True:
            return self.__ReadChain()
        else:
            self.StartFile(Mode = "r")

    def EndFile(self):
        self.Notify("!!CLOSING FILE -> " + self._FileDirName)
        self._File.close()
        self._File = None
        self._RefSet = None
   
    def __Store(self, Name, key, val):

        if isinstance(val, dict):
            self.__StoreDict(Name, key, val)

        elif isinstance(val, list):
            if len(val) == 0:
                self.__CreateAttribute(Name, key, val)
            else:
                self.__StoreList(Name, key, val)
        elif "AnalysisTopGNN" in str(type(val)):
            self.__CreateAttribute(Name, key, [str(val)])

        elif "torch.Tensor" in str(type(val)):
            self.__CreateAttribute(Name, key, val.numpy())

        elif "torch_geometric" in str(type(val)):
            self.__Store(Name, key, val.to_dict()) 

        elif "torch." in str(type(val)):
            self.__Store(Name, key, "")
        
        elif "type" in str(type(val)):
            pass

        else:
            self.__CreateAttribute(Name, key, val)

    def __CreateDataSet(self, RefName):
        self._RefSet.attrs[RefName] = self._File.create_dataset(RefName, data = h5py.Empty(None)).ref
    
    def __CreateAttribute(self, RefName, AttributeName, Data):
        if AttributeName in self._File[RefName].attrs:
            D = self._File[RefName].attrs[AttributeName]

            if isinstance(D, np.ndarray):
                D = np.append(D, Data)
            else: 
                D = [D, Data]
            self._File[RefName].attrs[AttributeName] = D
        else:
            self._File[RefName].attrs[AttributeName] = Data
        return self._File[RefName].attrs[AttributeName]

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
        
        if Name in self._RefSet.attrs:
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

        obj = self._File
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
