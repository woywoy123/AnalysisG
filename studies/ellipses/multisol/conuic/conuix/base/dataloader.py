from conuix.types.particle import Particle

class event:
    def __init__(self, idx):
        self.idx = idx
        self.strings = []
        self.bquark = []
        self.lepton = []
        self.neutrino = []
        self.wboson = []
        self.top = []

    def build_object(self, data, obj):
        bq = {}; x = -1; key = None
        for i in data:
            if obj in i: x+=1; bq[x] = {}; continue
            try: f = float(i)
            except: key = i.split(":")[0]; continue
            bq[x][key] = f
        return {i : Particle(bq[i]["px"], bq[i]["py"], bq[i]["pz"], bq[i]["e"]) for i in bq}

    def compile(self):
        ls = None
        for i in self.strings:
            if "bquark" in i: ls = self.bquark
            if "lepton" in i: ls = self.lepton
            if "neutrino" in i: ls = self.neutrino
            if "----" in i or "====" in i: continue
            if ls is None: continue
            ls.append(i)
        self.bquark   = self.build_object(self.bquark,     "bquark")
        self.lepton   = self.build_object(self.lepton,     "lepton")
        self.neutrino = self.build_object(self.neutrino, "neutrino")
        self.wboson   = {i : (self.lepton[i] + self.neutrino[i]) for i in self.lepton}
        self.top      = {i : (self.wboson[i] + self.bquark[i])   for i in self.wboson}

class DataLoader:
    def __init__(self, pth = "./conuix/data.txt"):
        self.event = {}
        k = -1
        f = open(pth).readlines()
        for i in f:
            data = i.split("\n")[0]
            for j in data.split(" "):
                if not len(j): continue
                if "EVENT" in j: k+=1; self.event[k] = event(k); continue
                try: self.event[k].strings.append(j)
                except: pass
        for i in self.event: self.event[i].compile()

    def __iter__(self): 
        self.itr = iter(self.event.values())
        return self

    def __next__(self):
        return next(self.itr)
