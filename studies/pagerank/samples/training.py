from AnalysisG.core.io import IO
import shutil 
import pickle
import random
import os

class lines:

    def __init__(self, data):
        self.data = data.split(":")
        self.prc = self.data[1].replace(", Post-Cut", "").replace(" ", "")
        self.PoC = float(self.data[2].replace(", Pre-Cut", "").replace(" ", ""))
        self.PrC = float(self.data[3].replace(", DatasetName", "").replace(" ", ""))
        self.Dset = self.data[4].replace(", x-section", "").replace(" ", "")
        self.xsec = self.data[5].replace(", directory", "").replace(" ", "")
        self.pth  = self.data[6].replace("\n","").replace(" ", "")
        self.eff = (self.PoC/self.PrC)
        self.cpgn = None
        self.scl = 0

class collect:

    def __init__(self, prc):
        self.prc = prc
        self.target = 0
        self.total = 0
        self.lins = []
    
    def __len__(self):
        return int(sum([i.PoC for i in self.lins]))
    
class Splitting:
    def __init__(self, root_path):
        self.root_path = root_path
        self.data_log = []
        self.training = {}
        self.evaluation = {}

        for i in open("./data/full-log.txt", "r").readlines():
            if "Campaign" in i: ms = i.split(" ")[2]; continue
            l = lines(i)
            l.cpgn = ms
            self.data_log.append(l)

        try: self = pickle.load(open("./data/splits.pkl", "rb"))
        except: self.GenWeights()
        self.GetStats(self.training, "training")
        self.GetStats(self.evaluation, "evaluation")

    def GenWeights(self):
        mc = {}  
        for i in self.data_log:
            if i.cpgn not in mc: mc[i.cpgn] = []
            mc[i.cpgn] += [i]

        wgt = {i : sum([j.eff for j in mc[i]])**0.5 for i in mc}
        for i in wgt: 
            for k in mc[i]: k.scl = (k.eff / wgt[i]) * k.PoC

        prc = {}
        all_prc = {}
        for i in mc:
            prc[i] = {}
            for k in mc[i]: 
                if k.prc not in prc[i]: prc[i][k.prc] = []
                prc[i][k.prc] += [k]

            for k in prc[i]:
                nv = 0
                for p in prc[i][k]: nv += p.scl
                if k not in all_prc: all_prc[k] = 0
                all_prc[k] += nv

        self.prc = prc
        self.all_prc = all_prc
        self.all_smpl = {}

        for i in all_prc:
            for m in prc:
                if i not in prc[m]: continue
                smpls = prc[m][i]
                if i not in self.all_smpl: self.all_smpl[i] = []
                self.all_smpl[i] += smpls

        total = 0
        for k in self.all_smpl:
            evn  = self.all_prc[k]
            smpl = self.all_smpl[k]
            col = collect(k)
            i = 0
            while i < evn:
                random.shuffle(smpl)
                el = smpl.pop()
                col.lins += [el]
                i += el.PoC
            col.target = evn
            col.total = len(col)
            self.training[k] = col
            
            col = collect(k)
            col.lins = smpl
            col.total = len(col)
            col.target = evn 
            self.evaluation[k] = col

        print("Dumped Samples")
        f = open("./data/splits.pkl", "wb")
        pickle.dump(self, f)
        f.close()

    def GetStats(self, mode, mode_str):
        out = ""
        total = 0
        mode = {i : mode[i] for i in sorted(mode)}
        for i in mode.values():
            out += "Processes: " + i.prc + "\n" 
            out += "Final Statistics: " + str(i.total) + "\n"
            out += "Target Statistics: " + str(i.target) + "\n"
            out += "DatasetNames: \n"
            out += "\n".join([k.Dset for k in i.lins]) + "\n\n"
            out += "DataPaths: \n"
            out += "\n".join([k.pth for k in i.lins]) + "\n\n"
            total += i.total
            
            pths = "./data/samples/" + mode_str
            try: os.makedirs(pths)
            except: pass

            for k in i.lins:
                trgt = pths + "/" + k.Dset
                try: os.makedirs(trgt)
                except: pass

                trn = trgt + "/" + k.pth.split("/")[-1]
                if os.path.isfile(trn.rstrip(".root") + ".valid"): continue
                shutil.copy(k.pth, trn)

                io = IO(trn)
                io.MetaCachePath = "./meta"
                io.Trees  = "nominal"
                io.Leaves = "weight_mc"
                io.EnablePyAMI = True
                lx = len([list(i.values())[0] for i in io])  
                fx = open(trn.rstrip(".root") + ".valid", "w")
                fx.write(str(lx))
                fx.close()
                print("Finished Copying to: " + trn)

        out += "Total Statistics: " + str(total)
        f = open("./data/" + mode_str + ".txt", "w")
        f.write(out)
        f.close()
