from AnalysisG.core.io import IO
from samples.mapping import *
import shutil 
import pickle
import os

class Samples:
    
    def __init__(self, root_path, tree = ["nominal"], leaf = ["weight_mc"]):
        self.path = root_path
        self.tree = tree
        self.leaf = leaf
        os.makedirs("./data", exist_ok = True)

        self.instance = None
        try: self.instance = pickle.load(open("./data/cache.pkl", "rb"))
        except: self._scan()
        self.dump_stats()

    def _scan(self):
        self.data = {}
        for subdir, dirs, files in os.walk(self.path):
            if subdir == self.path: continue
            for f in files:
                io = IO(subdir + "/" + f)
                io.MetaCachePath = "./meta"
                io.Trees  = self.tree
                io.Leaves = self.leaf
                io.EnablePyAMI = True
                lx = len([list(i.values())[0] for i in io])  

                it = IO(subdir + "/" + f)
                it.MetaCachePath = "./meta"
                it.Trees  = "truth"
                it.Leaves = self.leaf
                it.EnablePyAMI = False
                lt = len([list(i.values())[0] for i in it])  

                if lx == 0: continue
                meta = list(io.MetaData().values())
                for k in meta:
                    prc = k.physicsShort
                    prc = mapping(prc)
                    kx = None
                    if "r9364"  in k.DatasetName: kx = "mc16a"
                    if "r10201" in k.DatasetName: kx = "mc16d"
                    if "r10724" in k.DatasetName: kx = "mc16e"
                    if kx is None: continue

                    if kx  not in self.data:     self.data[kx] = {}
                    if prc not in self.data[kx]: self.data[kx][prc] = []
                    self.data[kx][prc].append([
                        k.crossSection, lx, lt, k.DatasetName, subdir + "/" + f
                    ])
            f = open(".cache.pkl", "wb")
            pickle.dump(self, f)
            f.close()
        self.instance = self

    def dump_stats(self):
        data = self.instance.data
        dxc = {}
        for a in data:
            if a not in dxc: dxc[a] = {}
            for i in data[a]: 
                if i not in dxc[a]: dxc[a][i] = {"pre_cut" : 0, "post_cut" : 0}
                for k in range(len(data[a][i])):
                    dxc[a][i]["post_cut"]  += data[a][i][k][1]
                    dxc[a][i]["pre_cut"]   += data[a][i][k][2]

        stry = "========= Pre-Cut =========\n" 
        for i in dxc:
            strx = "Campaign: " + i + " process : events pre cut \n"
            for k in dxc[i]:
                strx += k + " : " + str(dxc[i][k]["pre_cut"]) + " \n"
            stry += strx
 
        stry += "\n========= Post-Cut =========\n" 
        for i in dxc:
            strx = "Campaign: " + i + "process : events post cut \n"
            for k in dxc[i]:
                strx += k + " : " + str(dxc[i][k]["post_cut"]) + " \n"
            stry += strx
       
        f = open("./data/statistics.txt", "w")
        f.write(stry)
        f.close()

        O = []
        for i in data: 
            O += ["-------- Campaign: " + i + " --------"]
            for k in data[i]:
                for l in data[i][k]:
                    out = ", ".join([
                            "process: " + k, 
                            "Post-Cut: " + str(l[1]), 
                            "Pre-Cut: " + str(l[2]), 
                            "DatasetName: " + str(l[3]), 
                            "x-section: " + str(l[0]), 
                            "directory: " + str(l[-1])
                        ])
                    O += [out]
        
        f = open("./data/full-log.txt", "w")
        f.write("\n".join(O))
        f.close()

 

                




