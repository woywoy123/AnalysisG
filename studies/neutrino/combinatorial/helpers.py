import math

def makeDict(inpt):     return {i : 0  for i in inpt}
def protect(inpt):      return None if not len(inpt) else inpt[0]
def percent(totl, rsd): return {i : 100*rsd[i] / totl[i] for i in totl}

def get_nu(inpt):   return None if inpt is None else protect([i for i in inpt if i.is_nu ])
def get_jet(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_b  ])
def get_ell(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_lep])
def get_pxt(inpt): 
    x = [i for i in inpt if i is not None]
    if len(x) < 2: return sum(x)
    return None

def sumx(inpt, ox, fx, nfx = None):
    ix = range(len(inpt))
    if nfx is None: 
        for i in ix: ox += fx(inpt[i])
        return ox

    for i in ix: 
        if not nfx(inpt[i]): continue
        ox += fx(inpt[i])
    return ox

def bar(p, sym): return sym if p > 0 else "\\bar{" + sym + "}"

def safe_pair(inpt):
    n1, n2 = inpt
    if len(n1) and len(n2):     return n1  , n2
    if len(n1) and not len(n2): return None, n2
    if not len(n1) and len(n2): return n1, None
    return None, None

def pdgid(p):
    if abs(p) == 11: return bar(p, "e")
    if abs(p) == 12: return bar(p, "\\nu_{e}")
    if abs(p) == 13: return bar(p, "\\mu")
    if abs(p) == 14: return bar(p, "\\nu_{\\mu}")
    if abs(p) == 15: return bar(p, "\\tau")
    if abs(p) == 16: return bar(p, "\\nu_{\\tau}")
    print("!!!!!!!!!", p); exit()


class sets:
    def __init__(self, b = None, l = None, nu = None, rnu = None):
        self.tru_b   = b
        self.tru_l   = l
        self.tru_nu  = nu
        self.rnu     = rnu

    @property
    def truthtop(self): return get_pxt([self.tru_b, self.tru_l, self.tru_nu])
    @property
    def recotop(self): return get_pxt([self.tru_b, self.tru_l, self.rnu])
    
    @property
    def truthW(self): return get_pxt([self.tru_l, self.tru_nu])
    @property
    def recoW(self): return get_pxt([self.tru_l, self.rnu])


class data:
    def __init__(self):
        self.swp_ls  = {}
        self.swp_bs  = {}
        self.swp_lb  = {}
        self.swp_alb = {}

class atomic:
    def __init__(self):
        self.n_correct    = 0 # <- correct matched
        self.n_b_swapped  = 0 # <- only bs are swapped
        self.n_l_swapped  = 0 # <- only leptons are swapped
        self.n_bl_swapped = 0 # <- both bs and leptons are swapped
        self.n_unmatched  = 0 # <- Unmatched or fake neutrino
        self.n_non_nunu   = 0 # <- triggered on a non-dilepton event

        self.n_tru_nu  = 0
        self.n_rec_nu  = 0

        self.num_sols   = 0
        self.merged_jet = 0

        self.truth_set  = []
        self.reco_set   = []

        self.sets       = {0 : None, 1 : None}
        self.swapped_bs = {0 : None, 1 : None}
        self.swapped_bl = {0 : None, 1 : None}
        self.swapped_ls = {0 : None, 1 : None}
        self.fake_nus   = {0 : None, 1 : None}

    def match(self, chi2_match): 
        nt1, nt2 = safe_pair(self.truth_set)
        nr1, nr2 = safe_pair(self.reco_set)

        # -------- count the neutrinos ------- #
        if nt1 is not None: self.n_tru_nu += 1 
        if nt2 is not None: self.n_tru_nu += 1
        if nr1 is not None: self.n_rec_nu += 1
        if nr2 is not None: self.n_rec_nu += 1
        if not self.n_rec_nu: return # all nus as lost if self.n_tru_nu > 0

        tru_nu1, tru_nu2 =  get_nu(nt1),  get_nu(nt2)
        tru_bq1, tru_bq2 = get_jet(nt1), get_jet(nt2)
        tru_lp1, tru_lp2 = get_ell(nt1), get_ell(nt2)
        self.merged_jet += int(0 if tru_bq1 is None or tru_bq2 is None else tru_bq1 == tru_bq2)
        self.num_sols   += len(nr1)

        # -------- check whether the algorithm triggered in a non-dilepton event ------- #
        chx = {nr1[x].chi2 + nr2[x].chi2 : (nr1[x], nr2[x]) for x in range(len(nr1))}
        mix = sorted(list(chx))

        if mix[0] < 0: self.n_non_nunu += 2
        if not chi2_match: rnu1, rnu2 = nr1[0], nr2[0]
        else: rnu1, rnu2 = (None, None) if protect([i for i in mix if i > 0]) is None else chx[mix[0]]
        if rnu1 is None or rnu2 is None: return 

        # ------ matched ----- #
        mb = rnu1.matched_bquark == 1 and rnu2.matched_bquark == 1
        ml = rnu1.matched_lepton == 1 and rnu2.matched_lepton == 1

        # ------ swapped ----- #
        sb = rnu1.matched_bquark == -1 and rnu2.matched_bquark == -1
        sl = rnu1.matched_lepton == -1 and rnu2.matched_lepton == -1

        # ------ perfect match ----- #
        if mb and ml: 
            self.n_correct += 2
            self.sets[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.sets[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)

        # ------ swapped lepton and bquarks ----- #
        elif sb and sl: 
            self.n_l_swapped += 1
            self.n_b_swapped += 1
            self.swapped_bl[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.swapped_bl[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)

        # ------ swapped bquarks but matching leptons ------ # 
        elif sb and ml: 
            self.n_b_swapped += 2
            self.swapped_bs[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.swapped_bs[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)

        # ------ swapped leptons but matching bquarks ------- #
        elif sl and mb: 
            self.n_l_swapped += 2
            self.swapped_ls[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.swapped_ls[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)

        # ----------------- fake neutrino ----------------- #
        else: 
            self.n_unmatched += 2
            self.fake_nus[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.fake_nus[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)


class container:
    def __init__(self):
        self.atomics = {
            "top_children" : [], "truthjet"   : [], 
            "jetchildren"  : [], "jetleptons" : []
        }

        self.chi2_atomics = {
            "top_children" : [], "truthjet"   : [], 
            "jetchildren"  : [], "jetleptons" : []
        }

    def add(self, tru, reco, name, chi2_matched):
        a = atomic()
        a.truth_set += tru
        a.reco_set  += reco
        if not chi2_matched: self.atomics[name] += [a]
        else: self.chi2_atomics[name] += [a]
        a.match(chi2_matched)
        a.truth_set = []
        a.reco_set  = []

    def get(self, name, chi2):
        if not chi2: datax = self.atomics[name]
        else:   datax = self.chi2_atomics[name]

        # continue here with mass 
        out = data()

















    @property
    def efficiency(self):
        def fxt(k):   return (k.n_tru_nu    == 2)*2
        def fxc(k):   return (k.n_correct   == 2)*2
        def fx_ls(k): return (k.n_l_swapped == 2)*2
        def fx_bs(k): return (k.n_b_swapped == 2)*2
        def fx_bl(k): return (k.n_b_swapped == 1 and k.n_l_swapped == 1)*2
        def fx_mg(k): return 2*(fx_ls(k) > 0 or fx_bs(k) > 0 or fx_bl(k) > 0 or fxc(k) > 0)

        total = makeDict(self.atomics)
        for i in self.atomics: total[i] = sumx(self.atomics[i], 0, fxt)

        correct = makeDict(self.atomics)
        swp_ls, swp_bs = makeDict(self.atomics), makeDict(self.atomics)
        swp_lb, all_bl = makeDict(self.atomics), makeDict(self.atomics)
        for i in self.atomics: 
            swp_ls[i]  = sumx(self.atomics[i], 0, fx_ls)
            swp_bs[i]  = sumx(self.atomics[i], 0, fx_bs)
            swp_lb[i]  = sumx(self.atomics[i], 0, fx_bl)
            all_bl[i]  = sumx(self.atomics[i], 0, fx_mg)
            correct[i] = sumx(self.atomics[i], 0, fxc)

        chi2_correct = makeDict(self.chi2_atomics)
        swp_lscx, swp_bscx = makeDict(self.atomics), makeDict(self.atomics)
        swp_lbcx, all_blcx = makeDict(self.atomics), makeDict(self.atomics)
        for i in self.chi2_atomics: 
            swp_lscx[i] = sumx(self.chi2_atomics[i], 0, fx_ls)
            swp_bscx[i] = sumx(self.chi2_atomics[i], 0, fx_bs)
            swp_lbcx[i] = sumx(self.chi2_atomics[i], 0, fx_bl)
            all_blcx[i] = sumx(self.chi2_atomics[i], 0, fx_mg)
            chi2_correct[i] = sumx(self.chi2_atomics[i], 0, fxc)

       
        # ---------- compute the mistrigger ------------ #
        def ftn(k): return (k.n_tru_nu != 2) * (k.n_non_nunu != 2)
        def ffp(k): return (k.n_non_nunu == 2) * (k.n_tru_nu != 2)

        fake_rate, fake_rate_ch = {}, {}
        for i in self.atomics:
            tn = sumx(self.atomics[i], 0, ftn)
            fp = sumx(self.atomics[i], 0, ffp)
            fake_rate[i] = fp / (fp + tn)

            tn = sumx(self.chi2_atomics[i], 0, ftn)
            fp = sumx(self.chi2_atomics[i], 0, ffp)
            fake_rate_ch[i] = fp / (fp + tn)


        return {
                "nominal" : {
                    "swapped-leptons" : percent(total, swp_ls), "swapped-bquark"  : percent(total, swp_bs), 
                    "swapped-lepbqrk" : percent(total, swp_lb), "all-combination" : percent(total, all_bl), 
                    "correct" : percent(total, correct), "FPR" : fake_rate
                },
               "chi2-fitted" : {
                    "swapped-leptons" : percent(total, swp_lscx), "swapped-bquark"  : percent(total, swp_bscx),
                    "swapped-lepbqrk" : percent(total, swp_lbcx), "all-combination" : percent(total, all_blcx), 
                    "correct" : percent(total, chi2_correct), "FPR" : fake_rate_ch
                }
        }
        
def compiler(sel, names):
    con = container()
    for k in range(len(names)):
        n, c = names[k]
        for i in sel: con.add(i.TruthTops[n], i.RecoTops[n], n, c)
    return con

