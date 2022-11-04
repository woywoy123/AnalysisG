import torch 
import h5py
import numpy as np
import os
import sys 

from AnalysisTopGNN.Tools import Tools
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Notification import IO
from AnalysisTopGNN.Generators.Settings import Settings

class HDF5(Settings, Tools, IO):

    def __init__(self):
        self.Caller = "HDF5"
        Settings.__init__(self)
    
    def Start(self, Name = False, Mode = "w"):
        self.Filename += self._ext if self.Filename.endswith(self._ext) == False else ""
        self._File = h5py.File(self.Filename, mode = Mode, track_order = True)
        
        if Mode == "w":
            self.__IncrementRefSet(Name)
    
    def __IncrementRefSet(self, Name = False):
        self._iter += 1
        name = Name if Name else self._iter

        if str(name) in self._File:
            return 
        
        self.DumpingObjectName(Name)
        self._Ref = self._File.create_dataset(str(name), (1, ), dtype = h5py.ref_dtype)

    def __AddToDataSet(self, RefName, Key, Val = ""):
        if "AnalysisTopGNN" in str(type(Val)).split("'")[1]:
            if RefName not in self._Tracking:
                self._Tracking[RefName] = {}
            if str(hex(id(Val))) not in self._Tracking[RefName]:
                self._Tracking[RefName][str(hex(id(Val)))] = 0
            self._Tracking[RefName][str(hex(id(Val)))] += 1
            if self._Tracking[RefName][str(hex(id(Val)))] < 2:
                self.DumpObject(Val)
            Val = str(hex(id(Val)))
        elif "torch_geometric" in str(type(Val)).split("'")[1]:
            self.DumpObject(Val.to("cpu"))
            Val = str(hex(id(Val)))
        self._Ref.attrs[RefName + "." + Key] = Val

    def __Contains(self, key):
        return True if key in self._Ref.attrs else False

    def __Store(self, ObjPath, objectaddress, Key, Val):
            
            if self.__Contains(objectaddress) == False:
                self._Ref.attrs[objectaddress] = ObjPath
            
            if isinstance(Val, str):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, int):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, float):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, dict):
                for i in Val:
                    self.__AddToDataSet(objectaddress, Key + "-" + i, Val[i])
                return 
            elif isinstance(Val, list):
                for i in range(len(Val)):
                    self.__AddToDataSet(objectaddress, Key + "#" + str(i), Val[i]) 
                return 
            elif "Data" in ObjPath:
                if type(Val).__name__ == "type":
                    return 
                for i in list(Val):
                    self.__AddToDataSet(objectaddress, i, Val[i])
                return
            elif isinstance(Val, bool):
                self.__AddToDataSet(objectaddress, Key, Val)
                return 
            elif isinstance(Val, type(None)):
                return 
            elif isinstance(Val, np.int64):
                self.__AddToDataSet(objectaddress, Key, Val)
                return
            elif isinstance(Val, np.bool_):
                self.__AddToDataSet(objectaddress, Key, Val)
                return 
            print("NEED TO FIX THIS. COMING FROM HDF5", ObjPath, objectaddress, Key, Val, type(Val))

    def DumpObject(self, obj, Name = False):
        if self._iter == -1:
            self._Tracking = {}
            self.Start(Name = Name, Mode = "w")
        if Name:
            self._Tracking = {}
            self.__IncrementRefSet(Name)

        objname = str(type(obj)).split("'")[1]
        objectaddress = str(hex(id(obj)))
        for i in obj.__dict__:
            self.__Store(objname, objectaddress, i, obj.__dict__[i])
        return True
    
    def MultiThreadedDump(self, ObjectDict, OutputDirectory):

        def function(inpt):
            out = []
            for i in inpt:
                h = HDF5()
                h.VerboseLevel = self.VerboseLevel
                h.Filename = OutputDirectory + "/" + str(i[0])
                h.DumpObject(i[1], str(i[0]))
                out.append([h.Filename, str(i[0])]) 
            return out

        if isinstance(ObjectDict, dict) == False:
            self.WrongInputMultiThreading(ObjectDict)
            return 
        inpo = [[name, ObjectDict[name]] for name in ObjectDict]
        TH = Threading(inpo, function, self.Threads, self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
       
    def MergeHDF5(self, Directory):
        Files = self.DictToList(self.ListFilesInDir({Directory : ["*"]}, ".hdf5"))
        if len(Files) == 0:
            return 
        
        self.Filename = Directory + "/" + self.Filename
        self.Filename += self._ext if self.Filename.endswith(self._ext) == False else ""
        
        self._File = h5py.File(self.Filename, mode = "a", track_order = True)
        for i in self._File:
            for j in Files:
                if i in j:
                    Files.pop(Files.index(j))
                    break

        for i in Files:
            if i.endswith(self.Filename.split("/")[-1]):
                continue
            name = i.split("/")[-1].replace(self._ext, "")
            self._Ref = self._File.create_dataset(name, track_order = True, dtype = h5py.ref_dtype)
            src = h5py.File(i, mode = "r")
            self.MergingHDF5(i)
            for key in src:
                for attr in src[key].attrs:
                    self._Ref.attrs[attr] = src[key].attrs[attr]
            os.remove(i)
        self.End()

    def End(self):
        self._File.close()
        self.__init__()
    
    def __BuildContainer(self, obj, attr, i, typ):
        ins = {} if typ == "-" else None
        ins = [] if typ == "#" else ins
        
        attr = attr.split(typ)
        r = self._Ref.attrs[i]
        val = {attr[1] : r if r not in self._obj else self._obj[r][1]}

        if attr[0] not in obj.__dict__:
            setattr(obj, attr[0], ins)
        elif isinstance(obj.__dict__[attr[0]], type(ins)) == False:
            setattr(obj, attr[0], ins)

        v = getattr(obj, attr[0])
        if typ == "-":
            v |= val
        if typ == "#":
            setattr(obj, attr[0], v + list(val.values()))

    def RebuildObject(self, Name):
        self.Start(Mode = "r")
        self._Ref = None
        self._Ref = self._File[Name]
        
        objstruc = {n : self._Ref.attrs[n] for n in self._Ref.attrs}
        self._obj = {n : self.GetObjectFromString(".".join(objstruc[n].split(".")[:-1]), objstruc[n].split(".")[-1]) for n in objstruc if "." not in n}
        for i in self._obj:
            if self._obj[i][0] != None:
                ob = self._obj[i][1]()
                self._obj[i] = (None, ob)
        
        for i in self._Ref.attrs:
            val = i.split(".")
            
            # --- the variable "de" is the default value for an object. See if this causes a problem.
            de, obj = self._obj[val[0]]
            
            if len(val) == 1:
                continue
            attr = val[1]
            if "torch_geometric" in str(type(obj)).split("'")[1]:
                setattr(obj, attr, torch.tensor(self._Ref.attrs[i]))
                continue
            elif "-" in attr:
                self.__BuildContainer(obj, attr, i, "-")
                continue
            elif "#" in attr:
                self.__BuildContainer(obj, attr, i, "#")
                continue
            setattr(obj, attr, self._Ref.attrs[i])
        return [i[1] for i in self._obj.values() if "EventContainer" in str(type(i[1])).split("'")[1]][0]

    def __iter__(self):
        self.Start(Mode = "r")
        self._names = [i for i in self._File]
        return self

    def __next__(self):
        self._iter = -1
        if len(self._names) == 0:
            raise StopIteration()
        name = self._names.pop()
        return (name, self.RebuildObject(name))


