from utils import *

def penalty(err, tr, arr = [1]):
    a = (sum(err) - sum(tr)) ** 2
    b = sum(arr) ** 2
    return 1.0 / (1.0 + a / b) ** 0.5

def filtering(r):
    r_, t_, w_ = list(r.masses), list(r.truth), list(r.weight)
    if r.scoring.count("NOMINAL"): 
        msk = [i > 3.3 and i < 8 for i in r.weight]
        r_ = [j for i, j in zip(msk, r.masses) if i]
        w_ = [j for i, j in zip(msk, r.weight) if i]
    lr, lt = len(r_), len(t_)

    if not lr + lt:   return [r_, w_, penalty(r_, t_)]
    if not lt and lr: return [r_, w_, penalty(r_, t_)]
    if lt and not lr: return [r_, w_, penalty(r_, t_)]

    tp, tw = [], []
    while True:
        if not lr or not lt: return [tp, tw, penalty(r_, t_, tp)]
        sx = {(r_[i] - t_[j])**2 : [i, j] for i in range(lr) for j in range(lt)}
        mx = min(sorted(list(sx)))
        
        if mx >= 100:
            a = r_ + tp; b = w_ + tw
            return [a, b, penalty(r_, t_, tp if len(tp) else [1])]
        t_.pop(sx[mx][1]); 
        tp.append(r_.pop(sx[mx][0]))
        tw.append(w_.pop(sx[mx][0]))
        lr, lt = len(r_), len(t_)


class pairs:

    def __init__(self, ms = None, qlt = None):
        if isinstance(qlt, (float, int)) and isinstance(ms, (list, tuple)):
            self.mass = ms
            self.qual = qlt
        elif isinstance(ms, (float, int)) and isinstance(qlt, (list, tuple)):
            self.mass = qlt
            self.qual = ms
        else:
            self.mass = ms
            self.qual = qlt

    def __repr__(self):
        return f"pairs(mass={self.mass}, qual={self.qual})"

    def __getitem__(self, idx):
        if idx == 0: return self.mass
        if idx == 1: return self.qual
        raise IndexError(idx)

    def __iter__(self):
        yield self.mass
        yield self.qual

    def __len__(self):
        return 2

    @property
    def pred(self):
        return self.mass

    def __float__(self):
        return float(self.qual) if self.qual is not None else 0.0


class summary_t:

    def __init__(self, tru, msk):
        self.epoch     = str(tru.epoch)
        self.model     = str(tru.model)
        self.kfold     = str(tru.kfold)
        self.masked    = str(msk)

        self.predicted = {}
        self.pagerank  = {}
        self.quality   = {}
        self.truth     = {}
        self.matrix    = {i : {j : 0 for j in range(8)} for i in range(8)}
        self.nevents   = 0

    def __str__(self):
        return self.masked + "_" + self.model + "_" + self.epoch + "_" + self.kfold

class performance_t:

    def __init__(self):
        self.methods = {}

    def evaluate(self, mdl, tru, msk):
        smy = summary_t(tru, msk)
        if msk not in self.methods: self.methods[msk] = []
        self.methods[msk].append(smy)
        for r in mdl.expand(tru): 
            r.scoring = msk
            self.loop(r, smy)
        svpkl(smy, "cache/" + str(smy) + ".pkl") 

    def __check__(self, a, b, v):
        if a not in b: b[a] = []
        try: b[a] += v
        except: b[a] += [v]

    def loop(self, r, smy): 
        truth = r.truth
        out = filtering(r)
        pred, prg, qual = out[0], out[1], out[2]
        prc = r.prc
        if prc not in smy.quality: smy.quality[prc] = []
        if len(pred):
            smy.quality[prc].append(pairs(pred, qual))
        pred  = len(pred)  if len(pred)  < 5 else 5
        truth = len(truth) if len(truth) < 5 else 5
        smy.matrix[pred][truth] += 1
        smy.nevents += 1

    def summary(self):
        def head(i): return "\n ======= [EPOCH: " + i.epoch + ", kFOLD: " + i.kfold + "] ======= \n"
        def line(i): 
            kc = {k : average(i.quality[k]) for k in i.quality}
            a  = "\n ================================================== \n" 
            a += " MODEL: " + i.model + " " + i.masked + "\n"
            for k in kc:
                b = to_str(k) + ":"
                b = b + "".join( [" "] * (64 - len(b)) )
                b = b + str(kc[k]["av"]) + " +- " + str(kc[k]["s2"])
                a += b + "\n" 
            return a
        a = ""
        for i in self.methods: 
            a += head(self.methods[i][-1])
            a += line(self.methods[i][-1])
        print(a)

class stats_t:
    def __init__(self):
        self.train = {}
        self.valid = {}

        self.rtop_train = {}
        self.rtop_valid = {}

        self.atop_train = {}
        self.atop_valid = {}

        self.tr_name = "=================[Training]================="
        self.va_name = "=================[Validation]==============="


    def adding(self, dicx, key, lst):
        if key not in dicx: dicx[key] = []
        dicx[key] += [lst]

    def add(self, dt):
        px = dt.kfold + " | " + dt.prc
        
        if dt.mode == "training":   self.adding(self.train, px, 1)
        if dt.mode == "validation": self.adding(self.valid, px, 1)

        if dt.mode == "training":   self.adding(self.rtop_train, px, len(dt.truth))
        if dt.mode == "training":   self.adding(self.atop_train, px, len(dt.masses))

        if dt.mode == "validation": self.adding(self.rtop_valid, px, len(dt.truth))
        if dt.mode == "validation": self.adding(self.atop_valid, px, len(dt.masses))


    def __str__(self):
        skx = summary_stat(self.train)
        out = self.tr_name
        for px in skx: out += px + " : " + "\n".join(skx[px]) + "\n"
        out += "\n\n"

        sky = summary_stat(self.valid)
        out += self.va_name
        for px in sky: out += px + " : " + "\n".join(sky[px]) + "\n"
        return out.replace("kfold", "\n<+>").replace("<+>", "kfold")


    def TopSummary(self, kfld = None, data_tr = {}, data_vl = {}):
        if kfld is None: 
            setx = list(data_tr) + list(data_vl)
            setx = [i.split("|")[0].replace(" ", "") for i in setx if "kfold" in i]
            setx = sorted(list(set(setx)))
            out = "" 
            for i in setx: 
                out += "=================== ["  + i + "] ===================\n"
                out += self.TopSummary(i, data_tr, data_vl) 
            return out + "\n"
       
        skx = summary_top(data_tr, kfld)
        out = self.tr_name 
        out += "> Training < : {ntops : nevents} \n"
        for s in sorted(list(skx)): 
            out += "> " + s + " < : "
            out += " {" + " ".join([n + " : " + str(skx[s][n]) + ", " for n in skx[s] ]) + "}, \n"

        skx = summary_top(data_vl, kfld)
        out += self.va_name 
        out += "> Validation < : {ntops : nevents} \n"
        for s in sorted(list(skx)):
            out += "> " + s + " < : "
            out += " {" + " ".join([n + " : " + str(skx[s][n]) + ", " for n in skx[s] ]) + "}, \n"
        
        out = out.replace(" ", "").replace(">", "> ").replace(",},", "}").split("\n")
        for i in range(len(out)):
            if out[i].startswith("="): continue
            ps = out[i].split("<:")
            p1 = ps[0].replace("$", "").replace("> ", "")
            try: p2 = eval(ps[1].replace(">", ""))
            except: continue

            p2 = as_str({k : p2[k] if k in p2 else 0 for k in range(6)})
            out[i] = p2 + "<:" + p1 + ":>" 
        return "\n".join(out)

