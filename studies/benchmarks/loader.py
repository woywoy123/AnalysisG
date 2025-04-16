from AnalysisG.core.tools import Tools
import pickle
import tqdm

class Mdx:
    def __init__(self, mu, sig, idx):
        self.mu = mu[-1]
        self.sig = sig[-1]
        self.idx = idx[-1]
    def __str__(self):
        return "mean: " + str(self.mu) + ", stdev: " + str(self.sig) + ", index: " + str(self.idx)

class data:
    def __init__(self, pth = None, dev = None):
        self.device = dev
        self.pth = pth
        self.idx = None
        self.fx  = None

        self.cpu        = {"cpu"             : None,     "sig(cpu)"         : None, "dx" : None} # default
        self.cuda_t     = {"cuda(t)"         : None, "sig(cuda(t))"         : None, "dx" : None} # tensor
        self.cuda_k     = {"cuda(k)"         : None, "sig(cuda(k))"         : None, "dx" : None} # kernel
        self.cpu_cuda_t = {"cpu/cuda(t)"     : None, "sig(cpu/cuda(t))"     : None, "dx" : None}
        self.cpu_cuda_k = {"cpu/cuda(k)"     : None, "sig(cpu/cuda(k))"     : None, "dx" : None}
        self.cuda_t_k   = {"cuda(t)/cuda(k)" : None, "sig(cuda(t)/cuda(k))" : None, "dx" : None}

    def __lt__(self, other): return self.idx > other.idx
    def __gt__(self, other): return self.idx < other.idx
    def __eq__(self, other): return self.idx == other.idx

    def interpret(self):
        pkx = {}
        lsx = ["mu", "sig", "idx"]
        tl = Tools()
        for k in tqdm.tqdm(tl.ls(self.pth + "/" + self.device + "/*", "pkl")): 
            datax = k.split("/")[-1]
            fxx   = k.split("/")[-2]
            dx = pickle.load(open(k, "rb"))
            if fxx not in pkx: pkx[fxx] = []
            pkx[fxx].append(data())
            pkx[fxx][-1].idx = dx["dx"][-1]
            pkx[fxx][-1].cpu        = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cpu       )})
            pkx[fxx][-1].cuda_t     = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cuda_t    )})
            pkx[fxx][-1].cuda_k     = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cuda_k    )})
            pkx[fxx][-1].cpu_cuda_t = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cpu_cuda_t)})
            pkx[fxx][-1].cpu_cuda_k = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cpu_cuda_k)})
            pkx[fxx][-1].cuda_t_k   = Mdx(**{k : dx[j] for k, j in zip(lsx, self.cuda_t_k  )})
        return {fxx : sorted(pkx[fxx], reverse = True) for fxx in pkx}

def construct(pth, devices):
    tl = Tools()
    pkt = {}

    
    try: pkt = pickle.load(open("data.pkl", "rb"))
    except: 
        for i in devices: 
            dt = data(pth, i).interpret()
            for fxx in dt:
                if fxx not in pkt: pkt[fxx] = {}
                if i not in pkt[fxx]: pkt[fxx][i] = []
                pkt[fxx][i] = dt[fxx]
            pickle.dump(pkt, open("data.pkl", "wb"))
    return pkt

def assignment(pkt):
    physics = [
        "m", "p", "mt", "beta", "m2", "p2", 
        "mt2", "beta2", "theta", "deltaR"
    ]

    matrix = [
        "dot", "cofactor", "determinant",
        "eigenvalue", "inverse"
    ]

    transform_cartesian = [
        "px", "py", "pz", "pxpypz", "pxpypze"
    ]

    transform_polar = [
        "eta", "phi", "pt", "ptetaphi", "ptetaphie"
    ]

    neutrino = [
        "basematrix"
    ]

    pktx = {
            "operators"           : {}, 
            "neutrino"            : {},
            "physics-polar"       : {}, 
            "physics-cartesian"   : {}, 
            "transform-polar"     : {},
            "transform-cartesian" : {}
    }
    for i in pkt:
        tg = i.split("_")
        fx = tg[0]
        frame = None if len(tg) == 2 else tg[1]
        mode  = tg[-1]

        key = ""
        if fx in physics:             key = "physics"
        if fx in transform_cartesian: key = "transform-cartesian"
        if fx in transform_polar:     key = "transform-polar"
        if fx in matrix:              key = "operators"
        if fx in neutrino:            key = "neutrino"

        if frame == "polar":     key += "-polar"
        if frame == "cartesian": key += "-cartesian"
        
        if   mode == "separate": mode = "separate"
        elif mode == "combined": mode = "combined"
        else: mode = None 
        
        if mode is None: pktx[key] |= {fx : pkt[i]}; continue
        if fx not in pktx[key]: pktx[key][fx] = {"separate" : {}, "combined" : {}}
        pktx[key][fx][mode] |= pkt[i]
    return pktx













