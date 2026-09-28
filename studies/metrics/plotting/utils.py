import pickle
import pathlib

def depkl(pth):
    return pickle.load(open(pth, "rb"))

def svpkl(obj, pth):
    pickle.dump(obj, open(pth, "wb"))

def to_str(inpt): 
    try: return inpt.decode("utf-8")
    except: return ""

def mk_pth(inpt):
    try: pathlib.Path(inpt).mkdir(parents = True, exist_ok = True)
    except: return False
    return True

def pth_fm(bds, inpt): return [ (bds + "/" + i.lstrip("/")).replace("//", "/") for i in inpt]
def sms_lst(a): return sum(a, [])

def unq_lst(a): return list(set(a))

def get_dk(a, k):
    try: return a[k]
    except KeyError: return []

def add_dk(a, k):
    try: a[k]
    except KeyError: a[k] = {}
    return a

def add_ls(a, k):
    try: a[k]; return a
    except KeyError: a[k] = []
    return a



def mk_str(a, b, dl = " "): return str(a) + ": " + str(b) + dl

def colors():
    return iter(sorted(list(set([
        "aqua", "orange", "green","blue","olive","teal","gold",
        "darkblue","lime","crimson","magenta","orchid",
        "sienna","salmon","chocolate", "navy", "plum", "indigo", 
        "violet", "dodgerblue", "slategray", "ivory"
    ]))))

def sort_hist(hst):
    dcs = {}
    for i in hst:
        c = sum(i.counts)
        if c not in dcs: dcs[c] = []
        dcs[c] += [i]
    return sms_lst([dcs[i] for i in sorted(dcs)])
