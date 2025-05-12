from .utils import *
import tqdm

class sets:
    def __init__(self, b = None, l = None, nu = None, rnu = None):
        self.tru_b   = b
        self.tru_l   = l
        self.tru_nu  = nu
        self.rnu     = rnu
        self._truth_top = get_pxt([self.tru_b, self.tru_l, self.tru_nu])
        self._reco_top  = get_pxt([self.tru_b, self.tru_l, self.rnu])
        self._truth_W   = get_pxt([self.tru_l, self.tru_nu])
        self._reco_W    = get_pxt([self.tru_l, self.rnu])
        self._symbolic  = get_pdg([self.tru_l, self.tru_nu])

    @property
    def truthtop(self): return self._truth_top
    @property               
    def recotop(self):  return self._reco_top 
    @property          
    def truthW(self):   return self._truth_W  
    @property          
    def recoW(self):    return self._reco_W   
    @property         
    def symbolic(self): return self._symbolic 


class atomic:
    def __init__(self, as_dx = False):
        self.n_correct    = 0 if not as_dx else {} # <- correct matched
        self.n_b_swapped  = 0 if not as_dx else {} # <- only bs are swapped
        self.n_l_swapped  = 0 if not as_dx else {} # <- only leptons are swapped
        self.n_bl_swapped = 0 if not as_dx else {} # <- both bs and leptons are swapped
        self.n_unmatched  = 0 if not as_dx else {} # <- Unmatched or fake neutrino
        self.n_non_nunu   = 0 if not as_dx else {} # <- triggered on a non-dilepton event

        self.n_tru_nu  = 0 if not as_dx else {}
        self.n_rec_nu  = 0 if not as_dx else {}

        self.num_sols   = 0 if not as_dx else {}
        self.merged_jet = 0 if not as_dx else {}

        self.truth_set  = []
        self.reco_set   = []

        self.correct    = {0 : None if not as_dx else {}, 1 : None if not as_dx else {}} 
        self.swapped_bs = {0 : None if not as_dx else {}, 1 : None if not as_dx else {}} 
        self.swapped_bl = {0 : None if not as_dx else {}, 1 : None if not as_dx else {}} 
        self.swapped_ls = {0 : None if not as_dx else {}, 1 : None if not as_dx else {}} 
        self.fake_nus   = {0 : None if not as_dx else {}, 1 : None if not as_dx else {}} 

    def merge(self, _pdg, state, inpt):
        try: state[_pdg] += [inpt]
        except KeyError: state[_pdg] = [inpt]
        return state

    def __add__(self, other):
        if other == 0: return atomic(True).__add__(self)
        pdgx = extract(other.correct)
        pdgx = extract(other.swapped_bs) if pdgx is None else pdgx
        pdgx = extract(other.swapped_bl) if pdgx is None else pdgx
        pdgx = extract(other.swapped_ls) if pdgx is None else pdgx
        pdgx = extract(other.fake_nus)   if pdgx is None else pdgx
        if pdgx is None: return self

        for i in range(2):
            self.correct[i]       = self.merge(pdgx, self.correct[i]      , other.correct[i])
            self.swapped_bs[i] = self.merge(pdgx, self.swapped_bs[i], other.swapped_bs[i])
            self.swapped_bl[i] = self.merge(pdgx, self.swapped_bl[i], other.swapped_bl[i])
            self.swapped_ls[i] = self.merge(pdgx, self.swapped_ls[i], other.swapped_ls[i])
            self.fake_nus[i]   = self.merge(pdgx, self.fake_nus[i]  , other.fake_nus[i])

        self.n_correct    = self.merge(pdgx, self.n_correct   , other.n_correct   )
        self.n_b_swapped  = self.merge(pdgx, self.n_b_swapped , other.n_b_swapped )
        self.n_l_swapped  = self.merge(pdgx, self.n_l_swapped , other.n_l_swapped )
        self.n_bl_swapped = self.merge(pdgx, self.n_bl_swapped, other.n_bl_swapped)
        self.n_unmatched  = self.merge(pdgx, self.n_unmatched , other.n_unmatched )
        self.n_non_nunu   = self.merge(pdgx, self.n_non_nunu  , other.n_non_nunu  )
        self.n_tru_nu     = self.merge(pdgx, self.n_tru_nu    , other.n_tru_nu    )
        self.n_rec_nu     = self.merge(pdgx, self.n_rec_nu    , other.n_rec_nu    )
        self.num_sols     = self.merge(pdgx, self.num_sols    , other.num_sols    )
        self.merged_jet   = self.merge(pdgx, self.merged_jet  , other.merged_jet  )
        return self

    def __radd__(self, other): return self.__add__(other)

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
        if tru_bq1 is None or tru_bq2 is None: return
        if tru_lp1 is None or tru_lp2 is None: return

        # ------ matched ----- #
        mb = rnu1.matched_bquark == 1 and rnu2.matched_bquark == 1
        ml = rnu1.matched_lepton == 1 and rnu2.matched_lepton == 1

        # ------ swapped ----- #
        sb = rnu1.matched_bquark == -1 and rnu2.matched_bquark == -1
        sl = rnu1.matched_lepton == -1 and rnu2.matched_lepton == -1

        # ------ perfect match ----- #
        if mb and ml: 
            self.n_correct += 2
            self.correct[0] = sets(tru_bq1, tru_lp1, tru_nu1, rnu1)
            self.correct[1] = sets(tru_bq2, tru_lp2, tru_nu2, rnu2)

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
        return sum(datax)

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
        for i in tqdm.tqdm(sel, desc = n): con.add(i.TruthTops[n], i.RecoTops[n], n, c)
    return con

