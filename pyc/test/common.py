import torch
import vector
import inspect
import json
device = "cpu"

def create_vector_cartesian(px, py, pz, e): return vector.obj(px = px, py = py, pz = pz, E = e)
def create_vector_polar(pt, eta, phi, e): return vector.obj(pt = pt, eta = eta, phi = phi, E = e)
def create_tensor_cpu_1d(inpt = [1., 2., 3., 4.]): return torch.tensor(inpt).view(1, -1).to(dtype = torch.float64)
def create_tensor_cpu_nd(dim): return torch.cat([create_tensor_cpu_1d*(i+1) for i in range(dim)], 0)
def rounder(inpt1, inpt2, d = 4):
    try: i1 = inpt1.view(-1)[0].item()
    except:
        try: i1 = inpt1[0]
        except: i1 = inpt1

    diff = (abs(i1 - inpt2) / inpt2)*100
    if round(diff, d) > 0: print("----> ", i1, inpt2, round(diff, d))
    return round(diff, d) == 0

def rounder_l(inpt1, inpt2, d = 1):
    try: inpt1 = inpt1.view(-1).tolist()
    except:
        if isinstance(inpt1[0], list): inpt1 = inpt1[0]
        else: pass

    ok = True
    for i in range(len(inpt2)): ok *= rounder(inpt1[i], inpt2[i], d)
    return ok

def AttestEqual(truth, custom, threshold = 10**-10):
    x = abs(truth - custom) < threshold
    t = x.view(-1).sum(-1) == truth.view(-1).size(0)
    if not t:
        print(truth)
        print(custom)
        print((custom - truth)[x == False])
    return t

def compare(truth, pred, tolerance = 10**-6):
    mskt = truth == 0
    mskp = pred == 0
    ok = (mskt == mskp).view(-1)
    if not ok.sum(-1):
        print(truth)
        print(pred)
        print(pred - truth)
        return False

    x = abs(truth - pred)[mskt == False] / truth[mskt == False]
    x = (x > tolerance).view(-1)
    x = x.sum(-1) == 0
    if not x:
        print(x)
        print(truth)
        print(pred)
        mdk = abs(pred - truth) > tolerance
        print(pred[mdk], truth[mdk])
        print((pred - truth)[mdk])
        return False
    return x

class Particle:
    def __init__(self, pt, eta, phi, e):
        self.pt  = pt
        self.eta = eta
        self.phi = phi
        self.e   = e
        self._cuda = torch.cuda.is_available()

    @property
    def vec(self):
        dct = {
                "pt" : self.pt,
                "eta" : self.eta,
                "phi" : self.phi,
                "energy" : self.e
        }
        return vector.obj(**dct)

    @property
    def ten(self):
        vec = [self.pt, self.eta, self.phi, self.e]
        vec = torch.tensor([vec], dtype = torch.float64, device = self.cuda)
        return vec

    @property
    def ten_polar(self):
        return self.ten

    @property
    def ten_cart(self):
        return pyc.Transform.PxPyPzE(self.ten)

    @property
    def cuda(self):
        try: return "cuda" if self._cuda else "cpu"
        except: self._cuda = torch.cuda.is_available()
        return "cuda" if self._cuda else "cpu"

    @cuda.setter
    def cuda(self, v):
        self._cuda = v

class event:
    def __init__(self, met, phi):
        self.met = met
        self.phi = phi
        self._cuda = torch.cuda.is_available()

    @property
    def vec(self):
        return vector.obj(pt = self.met, phi = self.phi)

    @property
    def ten(self):
        vec = torch.tensor([self.vec.px, self.vec.py], dtype = torch.float64, device = self.cuda).view(-1, 2)
        return vec

    @property
    def ten_cart(self):
        return self.ten

    @property
    def ten_polar(self):
        vec = torch.tensor([self.met, self.phi], dtype = torch.float64, device = self.cuda).view(-1, 2)
        vec = vec.to(device = self.cuda)
        return vec

    @property
    def cuda(self):
        try: return "cuda" if self._cuda else "cpu"
        except: self._cuda = torch.cuda.is_available()
        return "cuda" if self._cuda else "cpu"

    @cuda.setter
    def cuda(self, v):
        self._cuda = v

def loads( n ):
    f = open(n + "NeutrinoEvents.json", "r")
    x = json.load(f)
    f.close()
    return [[event(**i[0])]+[Particle(**k) for k in i[1:]] for i in x]

def loadDouble(): return loads("Double")
def loadSingle(): return loads("Single")

