def bar(p, sym): return sym if p > 0 else "\\bar{" + sym + "}"
def pdgid(p):
    if abs(p) == 11: return bar(p, "e")
    if abs(p) == 12: return bar(p, "\\nu_{e}")
    if abs(p) == 13: return bar(p, "\\mu")
    if abs(p) == 14: return bar(p, "\\nu_{\\mu}")
    if abs(p) == 15: return bar(p, "\\tau")
    if abs(p) == 16: return bar(p, "\\nu_{\\tau}")
    print("!!!!!!!!!", p); exit()

class atomic:
    def __init__(self, t, particles):
        self.type    = t
        self.top1    = None
        self.top2    = None
        self.W1      = None
        self.W2      = None
        self.chi2_n1 = None
        self.chi2_n2 = None
        self.symbolics = (particles[0][1].pdgid, particles[1][1].pdgid)  if t == "truth" else None

        if particles[0][0] is None: return
        self.top1 = sum(particles[0]).Mass / 1000.0
        self.top2 = sum(particles[1]).Mass / 1000.0
        self.W1   = sum(particles[0][:2]).Mass / 1000.0
        self.W2   = sum(particles[1][:2]).Mass / 1000.0
        self.chi2_n1 = particles[0][0].chi2 if t != "truth" else 0
        self.chi2_n2 = particles[1][0].chi2 if t != "truth" else 0
        self.distance = particles[0][0].distance if t != "truth" else 0
   
    def __str__(self):
        out = "-----------" + str(self.type) + " -------------- \n"
        out += "t1: " + str(self.top1) + " "
        out += "t2: " + str(self.top2) + " \n"
        out += "w1: " + str(self.W1) + " "
        out += "w2: " + str(self.W2) + " \n"
        out += "chi2 (nu1): " + str(self.chi2_n1) + " "
        out += "chi2 (nu2): " + str(self.chi2_n2) + " \n"
        out += "symbolic: " + str(self.symbolics) + " "
        return out

class Container:
    def __init__(self, truths, cuda_dyn, cuda_st, ref_dyn, ref_st):
        self.truth    = atomic("truth", truths)
        self.cuda_dnu = atomic("cuda-dynamic", cuda_dyn)
        self.cuda_snu = atomic("cuda-static" , cuda_st)
        self.ref_dnu  = atomic("ref-dynamic" , ref_dyn)
        self.ref_snu  = atomic("ref-static"  , ref_st)
        self.event_data = {"met" : 0, "phi" : 0}

    def __str__(self):
        out = self.truth.__str__() + "\n\n"
        out += self.cuda_dnu.__str__() + "\n"
        out += self.cuda_snu.__str__() + "\n"
        out += self.ref_dnu.__str__() + "\n"
        out += self.ref_snu.__str__() + "\n"
        return out

class nux:
    def __init__(self):
        self.top1_masses = []
        self.top2_masses = []
        self.w1_masses = []
        self.w2_masses = []
        self.chi2_nu1 = []
        self.chi2_nu2 = []
        self.distances = []

        self.target_top1 = []
        self.target_top2 = []

        self.event_data = {"met" : [], "phi" : []}
        self.symbols = {}
        self.rejected = 0
        self.solution = 0
        self.num = 0
    
    def add(self, atm, truth, eps, con):
        self.num += 1
        if atm.top1 is None: return
        self.solution += 1

        kx = True
        if atm.type == "truth": pass
        elif eps is None: pass
        else: kx = atm.chi2_n1 < eps and atm.chi2_n2 < eps
        self.rejected += not kx
        if not kx: return

        if con is not None:
            self.target_top1.append(con.truth.top1)
            self.target_top2.append(con.truth.top2)

        self.top1_masses.append(atm.top1)
        self.top2_masses.append(atm.top2)

        self.w1_masses.append(atm.W1)
        self.w2_masses.append(atm.W2)
        if atm.type != "truth":
            self.chi2_nu1.append(atm.chi2_n1)
            self.chi2_nu2.append(atm.chi2_n2)
            self.distances.append(atm.distance)

        if con is not None:
            self.event_data["met"].append(con.event_data["met"])
            self.event_data["phi"].append(con.event_data["phi"])

        if truth is None: return 
        key = "(" + ", ".join([pdgid(i) for i in truth.symbolics]) + ")"
        if key not in self.symbols: self.symbols[key] = nux()
        self.symbols[key].add(atm, None, eps, con)

    def __str__(self):
        out  = "Total: "    + str(self.num) + " | "
        out += "Rejected: " + str(self.rejected) + " | " 
        out += "Passed: "   + str(self.solution - self.rejected) + " | "
        out += "Eff (solutions / events) (%): " + str(100*round(float(self.solution) / float(self.num), 4)) + " | "
        out += "Eff (passed / events) (%): " + str(100* round(float(self.solution - self.rejected) / float(self.num), 4))
        return out

class loss:
    def __init__(self, epsilon = None):
        self.num = 0
        self.loss_component = 0
        self.eps = epsilon

        self.truth     = nux()
        self.cuda_dnu  = nux()
        self.cuda_snu  = nux()
        self.ref_dnu   = nux()
        self.ref_snu   = nux()

    def add(self, con):
        if con is None: self.loss_component += 1; return
        self.num += 1
        typ = ["truth", "cuda_dnu", "cuda_snu", "ref_dnu", "ref_snu"] 
        for x in typ: getattr(self, x).add(getattr(con, x), con.truth, self.eps, con)
       
    def __str__(self):
        out  = "Number: " + str(self.num) + " "
        out += "Lost Components:" + str(self.loss_component) + " \n"
        out += "----------- CUDA (Dynamic) -----------\n"      + self.cuda_dnu.__str__() + "\n"
        out += "----------- CUDA (Static) -----------\n"       + self.cuda_snu.__str__() + "\n"
        out += "----------- Reference (Dynamic) -----------\n" + self.ref_dnu.__str__()  + "\n"
        out += "----------- Reference (Static) -----------\n"  + self.ref_snu.__str__()  + "\n"
        return out



