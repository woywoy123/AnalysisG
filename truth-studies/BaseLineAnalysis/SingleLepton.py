from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

class Container(object):

    def __init__(self, wrk):
        self.WorkingPoint = wrk
        self.LeptonicTop = []
        self.ResidualTop = []
        self.Top1 = []
        self.Top2 = []
        self.Zprime = []
        self.SampleIndex = {}
        self.SampleLumi = {}
    
    def Add(self, t1, t2, t3, t4, Zp, idx):
        l = t1 + t2 + t3 + t4
        if len(l) == 0:
            return

        self.LeptonicTop += t1
        self.ResidualTop += t4
        self.Top1 += t2
        self.Top2 += t3
        self.Zprime += Zp
        self.SampleIndex[idx] = Zp

    def MakeAllMass(self):
        self.lepT = [t.CalculateMass() for t in self.LeptonicTop]
        self.resT = [t.CalculateMass() for t in self.ResidualTop]
        self.t1T = [t.CalculateMass() for t in self.Top1]
        self.t2T = [t.CalculateMass() for t in self.Top2]
        self.Zprime = [t.CalculateMass() for t in self.Zprime if isinstance(t, int) == False]
        
        self.SampleIndex = { idx : self.SampleIndex[idx][0].CalculateMass() for idx in self.SampleIndex if self.SampleIndex[idx] != [] if isinstance(self.SampleIndex[idx][0], int) == False}

    def CalculateLumi(self, LumiCon):
        Lum = LumiCon.SampleLumi
        lumi = sum([Lum[l] for l in self.SampleIndex])
        self.Lumi = lumi
        self.CrossSec = len(self.Zprime) / (lumi + 1)


    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        C = Container(self.WorkingPoint)
        C.LeptonicTop += self.LeptonicTop
        C.ResidualTop += self.ResidualTop
        C.Top1 += self.Top1
        C.Top2 += self.Top2
 
        C.LeptonicTop += other.LeptonicTop
        C.ResidualTop += other.ResidualTop
        C.Top1 += other.Top1
        C.Top2 += other.Top2
        return C




def SingleLeptonAnalysis(Containers, ev):
   
    event = ev.Trees["nominal"]
    
    jets = []
    lep = []
    for p in event.DetectorParticles:
        if p.Type == "jet":
            jets.append(p)
        else:
            lep.append(p)
    
    Truth = [i for t in event.TopPostFSR for j in t.Jets for i in j.Children if t.FromRes == 1]
    tmp = []
    for t in Truth:
        if t not in tmp:
            tmp.append(t)
    Containers["Truth"].Zprime.append(sum(tmp))
    Containers["Truth"].SampleIndex[ev.EventIndex] = [sum(tmp)]
    Containers["Truth"].SampleLumi[ev.EventIndex] = event.weightmc



    # Find the lepton's closest jets but b-tagged at different eff points 
    # lep + jet @ 85, 77, 70, 60, 0
    jet85_lep = []
    jet77_lep = []
    jet70_lep = []
    jet60_lep = []
    jet0_lep = []
    
    if len(lep) != 1:
        return
    
    l = lep[0]
    dr_lep_matrix = { l.DeltaR(j) : j for j in jets}
    dr_matrix = list(dr_lep_matrix)
    dr_matrix.sort()
    
    
    tag_sum = { l.DeltaR(j) : sum([j.isbtagged_DL1r_85, j.isbtagged_DL1r_77, j.isbtagged_DL1r_70, j.isbtagged_DL1r_60]) for j in jets }
    
    def IgnoreEmpty(lst):
        if len(lst) == 0:
            return []
        return lst[0]
    
    jet85_lep += IgnoreEmpty([[dr_lep_matrix[r]] for r in dr_matrix if tag_sum[r] == 4])
    jet77_lep += IgnoreEmpty([[dr_lep_matrix[r]] for r in dr_matrix if tag_sum[r] == 3])
    jet70_lep += IgnoreEmpty([[dr_lep_matrix[r]] for r in dr_matrix if tag_sum[r] == 2])
    jet60_lep += IgnoreEmpty([[dr_lep_matrix[r]] for r in dr_matrix if tag_sum[r] == 1])
    jet0_lep += IgnoreEmpty([[dr_lep_matrix[r]] for r in dr_matrix if tag_sum[r] == 0])
    
    # ======= This is the leptonic top ======= #
    jet85_lep += lep
    jet77_lep += lep
    jet70_lep += lep
    jet60_lep += lep
    jet0_lep += lep
    
    
    
    # ==== Use remaining jets without the leptonically matched jet - We ignore basically merged jets 
    def GetTwo(inpt):
        if len(inpt) != 2:
            return []
        if len(inpt) == 2:
            return [inpt[0], inpt[1]]
    
    Jet_pt_matrix = { j.pt : j for j in jets }
    Jet_pt = list(Jet_pt_matrix)
    Jet_pt.sort()
    
    pt_tag_sum = { j.pt : sum([j.isbtagged_DL1r_85, j.isbtagged_DL1r_77, j.isbtagged_DL1r_70, j.isbtagged_DL1r_60]) for j in jets }
    
    # Find two hardest b-tagged jets - both being of equal btagging values and not within jet<xxx>_lep
    # - Have all jets tagged @ 85 but with different btagging of leptonically matched jet
    jj_85_at_jl85 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet85_lep and pt_tag_sum[j] == 4])
    jj_85_at_jl77 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet77_lep and pt_tag_sum[j] == 4])
    jj_85_at_jl70 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet70_lep and pt_tag_sum[j] == 4])
    jj_85_at_jl60 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet60_lep and pt_tag_sum[j] == 4])
    jj_85_at_jl0  = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet0_lep and pt_tag_sum[j] == 4])
    
    
    # - Have all jets tagged @ 77 but with different btagging of leptonically matched jet
    jj_77_at_jl85 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet85_lep and pt_tag_sum[j] == 3])
    jj_77_at_jl77 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet77_lep and pt_tag_sum[j] == 3])
    jj_77_at_jl70 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet70_lep and pt_tag_sum[j] == 3])
    jj_77_at_jl60 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet60_lep and pt_tag_sum[j] == 3])
    jj_77_at_jl0  = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet0_lep and pt_tag_sum[j] == 3])
    
    # - Have all jets tagged @ 70 but with different btagging of leptonically matched jet
    jj_70_at_jl85 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet85_lep and pt_tag_sum[j] == 2])
    jj_70_at_jl77 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet77_lep and pt_tag_sum[j] == 2])
    jj_70_at_jl70 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet70_lep and pt_tag_sum[j] == 2])
    jj_70_at_jl60 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet60_lep and pt_tag_sum[j] == 2])
    jj_70_at_jl0  = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet0_lep and pt_tag_sum[j] == 2])
    
    # - Have all jets tagged @ 60 but with different btagging of leptonically matched jet
    jj_60_at_jl85 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet85_lep and pt_tag_sum[j] == 1])
    jj_60_at_jl77 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet77_lep and pt_tag_sum[j] == 1])
    jj_60_at_jl70 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet70_lep and pt_tag_sum[j] == 1])
    jj_60_at_jl60 = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet60_lep and pt_tag_sum[j] == 1])
    jj_60_at_jl0  = GetTwo([Jet_pt_matrix[j] for j in Jet_pt if Jet_pt_matrix[j] not in jet0_lep and pt_tag_sum[j] == 1])
    
    # btagging @ 0 is ignored. Also for now we ignore jj_<xxx> where j1 != j2 btagging are different - could be another region
    
    # ===== Now find the closest jets to the selection above, making sure the two jets are in close proximity ====== #
    # -> This assumes that neither of the two jets are shared in the formed triplets
    def GetTriplet(lst, jlep):
        if len(lst) != 2:
            return [], [], [], []
    
        # -------- Non leptonically inclusion ---------#
        dic_j1 = {j.DeltaR(lst[0]) : j for j in jets if j != lst[0] and j not in jlep} # means dont include self we add this one later and the jet is not within the leptonic list
        dic_j1_dr = list(dic_j1)
        dic_j1_dr.sort()
        j1_lst = [dic_j1[j] for p, j in zip(range(len(dic_j1_dr)), dic_j1_dr) if p < 2] # means return the TWO cloests particles, hence p < 2 (0, 1)
        j1_lst += [lst[0]] # Add this one to the set, now we have a triplet 
    
        dic_j2 = {j.DeltaR(lst[1]) : j for j in jets if j != lst[1] and j not in jlep} # means dont include self we add this one later and the jet is not within the leptonic list
        dic_j2_dr = list(dic_j2)
        dic_j2_dr.sort()
        j2_lst = [dic_j2[j] for p, j in zip(range(len(dic_j2_dr)), dic_j2_dr) if p < 2] # means return the TWO cloests particles, hence p < 2 (0, 1)
        j2_lst += [lst[1]] # Add this one to the set, now we have a triplet 
    
        # -------- possible leptonic inclusion ---------#
        dic_j1 = {j.DeltaR(lst[0]) : j for j in jets if j != lst[0] and j} # means dont include self we add this one later
        dic_j1_dr = list(dic_j1)
        dic_j1_dr.sort()
        j1_lst_l = [dic_j1[j] for p, j in zip(range(len(dic_j1_dr)), dic_j1_dr) if p < 2] # means return the TWO cloests particles, hence p < 2 (0, 1)
        j1_lst_l += [lst[0]] # Add this one to the set, now we have a triplet 
    
        dic_j2 = {j.DeltaR(lst[1]) : j for j in jets if j != lst[1] and j} # means dont include self we add this one later
        dic_j2_dr = list(dic_j2)
        dic_j2_dr.sort()
        j2_lst_l = [dic_j2[j] for p, j in zip(range(len(dic_j2_dr)), dic_j2_dr) if p < 2] # means return the TWO cloests particles, hence p < 2 (0, 1)
        j2_lst_l += [lst[1]] # Add this one to the set, now we have a triplet 
    
        return j1_lst, j1_lst_l, j2_lst, j2_lst_l 
    
    
    def GetResiudalJets(inpt):
        return [j for j in jets if j not in inpt]
    
    
    # ======= These are now the resonance tops, highest pt tops ========== #
    # - jet1 and jet2 tagged at 85 but leptonically matched jet is btagged at different working points
    j1_85_at_njl85, j1_85_at_jl85, j2_85_at_njl85, j2_85_at_jl85 = GetTriplet(jj_85_at_jl85, jet85_lep) 
    j1_85_at_njl77, j1_85_at_jl77, j2_85_at_njl77, j2_85_at_jl77 = GetTriplet(jj_85_at_jl77, jet77_lep)
    j1_85_at_njl70, j1_85_at_jl70, j2_85_at_njl70, j2_85_at_jl70 = GetTriplet(jj_85_at_jl70, jet70_lep)
    j1_85_at_njl60, j1_85_at_jl60, j2_85_at_njl60, j2_85_at_jl60 = GetTriplet(jj_85_at_jl60, jet60_lep)
    j1_85_at_njl0 , j1_85_at_jl0 , j2_85_at_njl0 , j2_85_at_jl0  = GetTriplet(jj_85_at_jl0 , jet0_lep)
    
    # - jet1 and jet2 tagged at 77 but leptonically matched jet is btagged at different working points
    j1_77_at_njl85, j1_77_at_jl85, j2_77_at_njl85, j2_77_at_jl85 = GetTriplet(jj_77_at_jl85, jet85_lep) 
    j1_77_at_njl77, j1_77_at_jl77, j2_77_at_njl77, j2_77_at_jl77 = GetTriplet(jj_77_at_jl77, jet77_lep)
    j1_77_at_njl70, j1_77_at_jl70, j2_77_at_njl70, j2_77_at_jl70 = GetTriplet(jj_77_at_jl70, jet70_lep)
    j1_77_at_njl60, j1_77_at_jl60, j2_77_at_njl60, j2_77_at_jl60 = GetTriplet(jj_77_at_jl60, jet60_lep)
    j1_77_at_njl0 , j1_77_at_jl0 , j2_77_at_njl0 , j2_77_at_jl0  = GetTriplet(jj_77_at_jl0 , jet0_lep)
    
    # - jet1 and jet2 tagged at 70 but leptonically matched jet is btagged at different working points
    j1_70_at_njl85, j1_70_at_jl85, j2_70_at_njl85, j2_70_at_jl85 = GetTriplet(jj_70_at_jl85, jet85_lep) 
    j1_70_at_njl77, j1_70_at_jl77, j2_70_at_njl77, j2_70_at_jl77 = GetTriplet(jj_70_at_jl77, jet77_lep)
    j1_70_at_njl70, j1_70_at_jl70, j2_70_at_njl70, j2_70_at_jl70 = GetTriplet(jj_70_at_jl70, jet70_lep)
    j1_70_at_njl60, j1_70_at_jl60, j2_70_at_njl60, j2_70_at_jl60 = GetTriplet(jj_70_at_jl60, jet60_lep)
    j1_70_at_njl0 , j1_70_at_jl0 , j2_70_at_njl0 , j2_70_at_jl0  = GetTriplet(jj_70_at_jl0 , jet0_lep)
    
    # - jet1 and jet2 tagged at 60 but leptonically matched jet is btagged at different working points
    j1_60_at_njl85, j1_60_at_jl85, j2_60_at_njl85, j2_60_at_jl85 = GetTriplet(jj_60_at_jl85, jet85_lep) 
    j1_60_at_njl77, j1_60_at_jl77, j2_60_at_njl77, j2_60_at_jl77 = GetTriplet(jj_60_at_jl77, jet77_lep)
    j1_60_at_njl70, j1_60_at_jl70, j2_60_at_njl70, j2_60_at_jl70 = GetTriplet(jj_60_at_jl70, jet70_lep)
    j1_60_at_njl60, j1_60_at_jl60, j2_60_at_njl60, j2_60_at_jl60 = GetTriplet(jj_60_at_jl60, jet60_lep)
    j1_60_at_njl0 , j1_60_at_jl0 , j2_60_at_njl0 , j2_60_at_jl0  = GetTriplet(jj_60_at_jl0 , jet0_lep)
    
    
    # ===== Find the residual jets for each of the defined working points/regions to find the second spectator top ===== #
    # @ 85
    # Non inclusive leptonically matched jets 
    rej_85_at_njl85 = GetResiudalJets(j1_85_at_njl85 + j2_85_at_njl85 + jet85_lep)
    rej_85_at_njl77 = GetResiudalJets(j1_85_at_njl77 + j2_85_at_njl77 + jet77_lep)
    rej_85_at_njl70 = GetResiudalJets(j1_85_at_njl70 + j2_85_at_njl70 + jet70_lep)
    rej_85_at_njl60 = GetResiudalJets(j1_85_at_njl60 + j2_85_at_njl60 + jet60_lep)
    rej_85_at_njl0  = GetResiudalJets(j1_85_at_njl0  + j2_85_at_njl0  + jet0_lep)
    
    # Inclusive leptonically matched jets 
    rej_85_at_jl85 = GetResiudalJets(j1_85_at_jl85 + j2_85_at_jl85 + jet85_lep)
    rej_85_at_jl77 = GetResiudalJets(j1_85_at_jl77 + j2_85_at_jl77 + jet77_lep)
    rej_85_at_jl70 = GetResiudalJets(j1_85_at_jl70 + j2_85_at_jl70 + jet70_lep)
    rej_85_at_jl60 = GetResiudalJets(j1_85_at_jl60 + j2_85_at_jl60 + jet60_lep)
    rej_85_at_jl0  = GetResiudalJets(j1_85_at_jl0  + j2_85_at_jl0  + jet0_lep)
    
    # @ 77
    # Non inclusive leptonically matched jets 
    rej_77_at_njl85 = GetResiudalJets(j1_77_at_njl85 + j2_77_at_njl85 + jet85_lep)
    rej_77_at_njl77 = GetResiudalJets(j1_77_at_njl77 + j2_77_at_njl77 + jet77_lep)
    rej_77_at_njl70 = GetResiudalJets(j1_77_at_njl70 + j2_77_at_njl70 + jet70_lep)
    rej_77_at_njl60 = GetResiudalJets(j1_77_at_njl60 + j2_77_at_njl60 + jet60_lep)
    rej_77_at_njl0  = GetResiudalJets(j1_77_at_njl0  + j2_77_at_njl0  + jet0_lep)
    
    # Inclusive leptonically matched jets 
    rej_77_at_jl85 = GetResiudalJets(j1_77_at_jl85 + j2_77_at_jl85 + jet85_lep)
    rej_77_at_jl77 = GetResiudalJets(j1_77_at_jl77 + j2_77_at_jl77 + jet77_lep)
    rej_77_at_jl70 = GetResiudalJets(j1_77_at_jl70 + j2_77_at_jl70 + jet70_lep)
    rej_77_at_jl60 = GetResiudalJets(j1_77_at_jl60 + j2_77_at_jl60 + jet60_lep)
    rej_77_at_jl0  = GetResiudalJets(j1_77_at_jl0  + j2_77_at_jl0  + jet0_lep)
    
    # @ 70
    # Non inclusive leptonically matched jets 
    rej_70_at_njl85 = GetResiudalJets(j1_70_at_njl85 + j2_70_at_njl85 + jet85_lep)
    rej_70_at_njl77 = GetResiudalJets(j1_70_at_njl77 + j2_70_at_njl77 + jet77_lep)
    rej_70_at_njl70 = GetResiudalJets(j1_70_at_njl70 + j2_70_at_njl70 + jet70_lep)
    rej_70_at_njl60 = GetResiudalJets(j1_70_at_njl60 + j2_70_at_njl60 + jet60_lep)
    rej_70_at_njl0  = GetResiudalJets(j1_70_at_njl0  + j2_70_at_njl0  + jet0_lep)
    
    # Inclusive leptonically matched jets 
    rej_70_at_jl85 = GetResiudalJets(j1_70_at_jl85 + j2_70_at_jl85 + jet85_lep)
    rej_70_at_jl77 = GetResiudalJets(j1_70_at_jl77 + j2_70_at_jl77 + jet77_lep)
    rej_70_at_jl70 = GetResiudalJets(j1_70_at_jl70 + j2_70_at_jl70 + jet70_lep)
    rej_70_at_jl60 = GetResiudalJets(j1_70_at_jl60 + j2_70_at_jl60 + jet60_lep)
    rej_70_at_jl0  = GetResiudalJets(j1_70_at_jl0  + j2_70_at_jl0  + jet0_lep)
    
    # @ 60
    # Non inclusive leptonically matched jets 
    rej_60_at_njl85 = GetResiudalJets(j1_60_at_njl85 + j2_60_at_njl85 + jet85_lep)
    rej_60_at_njl77 = GetResiudalJets(j1_60_at_njl77 + j2_60_at_njl77 + jet77_lep)
    rej_60_at_njl70 = GetResiudalJets(j1_60_at_njl70 + j2_60_at_njl70 + jet70_lep)
    rej_60_at_njl60 = GetResiudalJets(j1_60_at_njl60 + j2_60_at_njl60 + jet60_lep)
    rej_60_at_njl0  = GetResiudalJets(j1_60_at_njl0  + j2_60_at_njl0  + jet0_lep)
    
    # Inclusive leptonically matched jets 
    rej_60_at_jl85 = GetResiudalJets(j1_60_at_jl85 + j2_60_at_jl85 + jet85_lep)
    rej_60_at_jl77 = GetResiudalJets(j1_60_at_jl77 + j2_60_at_jl77 + jet77_lep)
    rej_60_at_jl70 = GetResiudalJets(j1_60_at_jl70 + j2_60_at_jl70 + jet70_lep)
    rej_60_at_jl60 = GetResiudalJets(j1_60_at_jl60 + j2_60_at_jl60 + jet60_lep)
    rej_60_at_jl0  = GetResiudalJets(j1_60_at_jl0  + j2_60_at_jl0  + jet0_lep)
    
    
    # ===== Now calculate the top masses for each region ====== #
    # leptonic spec - resonance tops - residual spec
    def ReturnMass(lepT, T1, T2, resid):
        lt, t1, t2, res, Zp = sum(lepT), sum(T1), sum(T2), sum(resid), sum([t for t in T1 if t not in T2])
        
        lt = [lt] if isinstance(lt, int) == False else []
        t1 = [t1] if isinstance(t1, int) == False else []
        t2 = [t2] if isinstance(t2, int) == False else []
        res = [res] if isinstance(res, int) == False else []
        zp = [Zp] if isinstance(Zp, int) == False else []

        return lt, t1, t2, res, zp
    
    
    # @ 85
    # Non inclusive leptonic 
    top_lep_85_85, top_j1_85_85, top_j2_85_85, top_rej_85_85, Zprime_85_85 = ReturnMass(jet85_lep, j1_85_at_njl85, j2_85_at_njl85, rej_85_at_njl85)
    top_lep_85_77, top_j1_85_77, top_j2_85_77, top_rej_85_77, Zprime_85_77 = ReturnMass(jet77_lep, j1_85_at_njl77, j2_85_at_njl77, rej_85_at_njl77)
    top_lep_85_70, top_j1_85_70, top_j2_85_70, top_rej_85_70, Zprime_85_70 = ReturnMass(jet70_lep, j1_85_at_njl70, j2_85_at_njl70, rej_85_at_njl70)
    top_lep_85_60, top_j1_85_60, top_j2_85_60, top_rej_85_60, Zprime_85_60 = ReturnMass(jet60_lep, j1_85_at_njl60, j2_85_at_njl60, rej_85_at_njl60)
    top_lep_85_0 , top_j1_85_0 , top_j2_85_0 , top_rej_85_0 , Zprime_85_0  = ReturnMass(jet0_lep , j1_85_at_njl0 , j2_85_at_njl0 , rej_85_at_njl0 )
    
    # Inclusive leptonic 
    i_top_lep_85_85, i_top_j1_85_85, i_top_j2_85_85, i_top_rej_85_85, i_Zprime_85_85  = ReturnMass(jet85_lep, j1_85_at_jl85, j2_85_at_jl85, rej_85_at_jl85)
    i_top_lep_85_77, i_top_j1_85_77, i_top_j2_85_77, i_top_rej_85_77, i_Zprime_85_77  = ReturnMass(jet77_lep, j1_85_at_jl77, j2_85_at_jl77, rej_85_at_jl77)
    i_top_lep_85_70, i_top_j1_85_70, i_top_j2_85_70, i_top_rej_85_70, i_Zprime_85_70  = ReturnMass(jet70_lep, j1_85_at_jl70, j2_85_at_jl70, rej_85_at_jl70)
    i_top_lep_85_60, i_top_j1_85_60, i_top_j2_85_60, i_top_rej_85_60, i_Zprime_85_60  = ReturnMass(jet60_lep, j1_85_at_jl60, j2_85_at_jl60, rej_85_at_jl60)
    i_top_lep_85_0 , i_top_j1_85_0 , i_top_j2_85_0 , i_top_rej_85_0 , i_Zprime_85_0   = ReturnMass(jet0_lep , j1_85_at_jl0 , j2_85_at_jl0 , rej_85_at_jl0 )
    
    
    # @ 77
    # Non inclusive leptonic 
    top_lep_77_85, top_j1_77_85, top_j2_77_85, top_rej_77_85, Zprime_77_85  = ReturnMass(jet85_lep, j1_77_at_njl85, j2_77_at_njl85, rej_77_at_njl85)
    top_lep_77_77, top_j1_77_77, top_j2_77_77, top_rej_77_77, Zprime_77_77  = ReturnMass(jet77_lep, j1_77_at_njl77, j2_77_at_njl77, rej_77_at_njl77)
    top_lep_77_70, top_j1_77_70, top_j2_77_70, top_rej_77_70, Zprime_77_70  = ReturnMass(jet70_lep, j1_77_at_njl70, j2_77_at_njl70, rej_77_at_njl70)
    top_lep_77_60, top_j1_77_60, top_j2_77_60, top_rej_77_60, Zprime_77_60  = ReturnMass(jet60_lep, j1_77_at_njl60, j2_77_at_njl60, rej_77_at_njl60)
    top_lep_77_0 , top_j1_77_0 , top_j2_77_0 , top_rej_77_0 , Zprime_77_0   = ReturnMass(jet0_lep , j1_77_at_njl0 , j2_77_at_njl0 , rej_77_at_njl0 )
    
    # Inclusive leptonic 
    i_top_lep_77_85, i_top_j1_77_85, i_top_j2_77_85, i_top_rej_77_85, i_Zprime_77_85  = ReturnMass(jet85_lep, j1_77_at_jl85, j2_77_at_jl85, rej_77_at_jl85)
    i_top_lep_77_77, i_top_j1_77_77, i_top_j2_77_77, i_top_rej_77_77, i_Zprime_77_77  = ReturnMass(jet77_lep, j1_77_at_jl77, j2_77_at_jl77, rej_77_at_jl77)
    i_top_lep_77_70, i_top_j1_77_70, i_top_j2_77_70, i_top_rej_77_70, i_Zprime_77_70  = ReturnMass(jet70_lep, j1_77_at_jl70, j2_77_at_jl70, rej_77_at_jl70)
    i_top_lep_77_60, i_top_j1_77_60, i_top_j2_77_60, i_top_rej_77_60, i_Zprime_77_60  = ReturnMass(jet60_lep, j1_77_at_jl60, j2_77_at_jl60, rej_77_at_jl60)
    i_top_lep_77_0 , i_top_j1_77_0 , i_top_j2_77_0 , i_top_rej_77_0 , i_Zprime_77_0   = ReturnMass(jet0_lep , j1_77_at_jl0 , j2_77_at_jl0 , rej_77_at_jl0 )
    
    
    # @ 70
    # Non inclusive leptonic 
    top_lep_70_85, top_j1_70_85, top_j2_70_85, top_rej_70_85, Zprime_70_85  = ReturnMass(jet85_lep, j1_70_at_njl85, j2_70_at_njl85, rej_70_at_njl85)
    top_lep_70_77, top_j1_70_77, top_j2_70_77, top_rej_70_77, Zprime_70_77  = ReturnMass(jet77_lep, j1_70_at_njl77, j2_70_at_njl77, rej_70_at_njl77)
    top_lep_70_70, top_j1_70_70, top_j2_70_70, top_rej_70_70, Zprime_70_70  = ReturnMass(jet70_lep, j1_70_at_njl70, j2_70_at_njl70, rej_70_at_njl70)
    top_lep_70_60, top_j1_70_60, top_j2_70_60, top_rej_70_60, Zprime_70_60  = ReturnMass(jet60_lep, j1_70_at_njl60, j2_70_at_njl60, rej_70_at_njl60)
    top_lep_70_0 , top_j1_70_0 , top_j2_70_0 , top_rej_70_0 , Zprime_70_0   = ReturnMass(jet0_lep , j1_70_at_njl0 , j2_70_at_njl0 , rej_70_at_njl0 )
    
    # Inclusive leptonic 
    i_top_lep_70_85, i_top_j1_70_85, i_top_j2_70_85, i_top_rej_70_85, i_Zprime_70_85  = ReturnMass(jet85_lep, j1_70_at_jl85, j2_70_at_jl85, rej_70_at_jl85)
    i_top_lep_70_77, i_top_j1_70_77, i_top_j2_70_77, i_top_rej_70_77, i_Zprime_70_77  = ReturnMass(jet77_lep, j1_70_at_jl77, j2_70_at_jl77, rej_70_at_jl77)
    i_top_lep_70_70, i_top_j1_70_70, i_top_j2_70_70, i_top_rej_70_70, i_Zprime_70_70  = ReturnMass(jet70_lep, j1_70_at_jl70, j2_70_at_jl70, rej_70_at_jl70)
    i_top_lep_70_60, i_top_j1_70_60, i_top_j2_70_60, i_top_rej_70_60, i_Zprime_70_60  = ReturnMass(jet60_lep, j1_70_at_jl60, j2_70_at_jl60, rej_70_at_jl60)
    i_top_lep_70_0 , i_top_j1_70_0 , i_top_j2_70_0 , i_top_rej_70_0 , i_Zprime_70_0   = ReturnMass(jet0_lep , j1_70_at_jl0 , j2_70_at_jl0 , rej_70_at_jl0 )
    
    
    # @ 60
    # Non inclusive leptonic 
    top_lep_60_85, top_j1_60_85, top_j2_60_85, top_rej_60_85, Zprime_60_85  = ReturnMass(jet85_lep, j1_60_at_njl85, j2_60_at_njl85, rej_60_at_njl85)
    top_lep_60_77, top_j1_60_77, top_j2_60_77, top_rej_60_77, Zprime_60_77  = ReturnMass(jet77_lep, j1_60_at_njl77, j2_60_at_njl77, rej_60_at_njl77)
    top_lep_60_70, top_j1_60_70, top_j2_60_70, top_rej_60_70, Zprime_60_70  = ReturnMass(jet70_lep, j1_60_at_njl70, j2_60_at_njl70, rej_60_at_njl70)
    top_lep_60_60, top_j1_60_60, top_j2_60_60, top_rej_60_60, Zprime_60_60  = ReturnMass(jet60_lep, j1_60_at_njl60, j2_60_at_njl60, rej_60_at_njl60)
    top_lep_60_0 , top_j1_60_0 , top_j2_60_0 , top_rej_60_0 , Zprime_60_0   = ReturnMass(jet0_lep , j1_60_at_njl0 , j2_60_at_njl0 , rej_60_at_njl0 )
    
    # Inclusive leptonic 
    i_top_lep_60_85, i_top_j1_60_85, i_top_j2_60_85, i_top_rej_60_85, i_Zprime_60_85  = ReturnMass(jet85_lep, j1_60_at_jl85, j2_60_at_jl85, rej_60_at_jl85)
    i_top_lep_60_77, i_top_j1_60_77, i_top_j2_60_77, i_top_rej_60_77, i_Zprime_60_77  = ReturnMass(jet77_lep, j1_60_at_jl77, j2_60_at_jl77, rej_60_at_jl77)
    i_top_lep_60_70, i_top_j1_60_70, i_top_j2_60_70, i_top_rej_60_70, i_Zprime_60_70  = ReturnMass(jet70_lep, j1_60_at_jl70, j2_60_at_jl70, rej_60_at_jl70)
    i_top_lep_60_60, i_top_j1_60_60, i_top_j2_60_60, i_top_rej_60_60, i_Zprime_60_60  = ReturnMass(jet60_lep, j1_60_at_jl60, j2_60_at_jl60, rej_60_at_jl60)
    i_top_lep_60_0 , i_top_j1_60_0 , i_top_j2_60_0 , i_top_rej_60_0 , i_Zprime_60_0   = ReturnMass(jet0_lep , j1_60_at_jl0 , j2_60_at_jl0 , rej_60_at_jl0 )

    

    Containers["top_85_85"].Add(top_lep_85_85, top_j1_85_85, top_j2_85_85, top_rej_85_85, Zprime_85_85, ev.EventIndex)
    Containers["top_85_77"].Add(top_lep_85_77, top_j1_85_77, top_j2_85_77, top_rej_85_77, Zprime_85_77, ev.EventIndex)
    Containers["top_85_70"].Add(top_lep_85_70, top_j1_85_70, top_j2_85_70, top_rej_85_70, Zprime_85_70, ev.EventIndex)
    Containers["top_85_60"].Add(top_lep_85_60, top_j1_85_60, top_j2_85_60, top_rej_85_60, Zprime_85_60, ev.EventIndex)
    Containers["top_85_0" ].Add(top_lep_85_0 , top_j1_85_0 , top_j2_85_0 , top_rej_85_0 , Zprime_85_0 , ev.EventIndex)
                                                                                            
    Containers["top_77_85"].Add(top_lep_77_85, top_j1_77_85, top_j2_77_85, top_rej_77_85, Zprime_77_85, ev.EventIndex)
    Containers["top_77_77"].Add(top_lep_77_77, top_j1_77_77, top_j2_77_77, top_rej_77_77, Zprime_77_77, ev.EventIndex)
    Containers["top_77_70"].Add(top_lep_77_70, top_j1_77_70, top_j2_77_70, top_rej_77_70, Zprime_77_70, ev.EventIndex)
    Containers["top_77_60"].Add(top_lep_77_60, top_j1_77_60, top_j2_77_60, top_rej_77_60, Zprime_77_60, ev.EventIndex)
    Containers["top_77_0" ].Add(top_lep_77_0 , top_j1_77_0 , top_j2_77_0 , top_rej_77_0 , Zprime_77_0 , ev.EventIndex)
                                                                                            
    Containers["top_70_85"].Add(top_lep_70_85, top_j1_70_85, top_j2_70_85, top_rej_70_85, Zprime_70_85, ev.EventIndex)
    Containers["top_70_77"].Add(top_lep_70_77, top_j1_70_77, top_j2_70_77, top_rej_70_77, Zprime_70_77, ev.EventIndex)
    Containers["top_70_70"].Add(top_lep_70_70, top_j1_70_70, top_j2_70_70, top_rej_70_70, Zprime_70_70, ev.EventIndex)
    Containers["top_70_60"].Add(top_lep_70_60, top_j1_70_60, top_j2_70_60, top_rej_70_60, Zprime_70_60, ev.EventIndex)
    Containers["top_70_0" ].Add(top_lep_70_0 , top_j1_70_0 , top_j2_70_0 , top_rej_70_0 , Zprime_70_0 , ev.EventIndex)
                                                                                          
    Containers["top_60_85"].Add(top_lep_60_85, top_j1_60_85, top_j2_60_85, top_rej_60_85, Zprime_60_85, ev.EventIndex)
    Containers["top_60_77"].Add(top_lep_60_77, top_j1_60_77, top_j2_60_77, top_rej_60_77, Zprime_60_77, ev.EventIndex)
    Containers["top_60_70"].Add(top_lep_60_70, top_j1_60_70, top_j2_60_70, top_rej_60_70, Zprime_60_70, ev.EventIndex)
    Containers["top_60_60"].Add(top_lep_60_60, top_j1_60_60, top_j2_60_60, top_rej_60_60, Zprime_60_60, ev.EventIndex)
    Containers["top_60_0" ].Add(top_lep_60_0 , top_j1_60_0 , top_j2_60_0 , top_rej_60_0 , Zprime_60_0 , ev.EventIndex)

    Containers["inc_top_85_85"].Add(i_top_lep_85_85, i_top_j1_85_85, i_top_j2_85_85, i_top_rej_85_85, i_Zprime_85_85, ev.EventIndex)
    Containers["inc_top_85_77"].Add(i_top_lep_85_77, i_top_j1_85_77, i_top_j2_85_77, i_top_rej_85_77, i_Zprime_85_77, ev.EventIndex)
    Containers["inc_top_85_70"].Add(i_top_lep_85_70, i_top_j1_85_70, i_top_j2_85_70, i_top_rej_85_70, i_Zprime_85_70, ev.EventIndex)
    Containers["inc_top_85_60"].Add(i_top_lep_85_60, i_top_j1_85_60, i_top_j2_85_60, i_top_rej_85_60, i_Zprime_85_60, ev.EventIndex)
    Containers["inc_top_85_0" ].Add(i_top_lep_85_0 , i_top_j1_85_0 , i_top_j2_85_0 , i_top_rej_85_0 , i_Zprime_85_0 , ev.EventIndex)
                                                                                                       
    Containers["inc_top_77_85"].Add(i_top_lep_77_85, i_top_j1_77_85, i_top_j2_77_85, i_top_rej_77_85, i_Zprime_77_85, ev.EventIndex)
    Containers["inc_top_77_77"].Add(i_top_lep_77_77, i_top_j1_77_77, i_top_j2_77_77, i_top_rej_77_77, i_Zprime_77_77, ev.EventIndex)
    Containers["inc_top_77_70"].Add(i_top_lep_77_70, i_top_j1_77_70, i_top_j2_77_70, i_top_rej_77_70, i_Zprime_77_70, ev.EventIndex)
    Containers["inc_top_77_60"].Add(i_top_lep_77_60, i_top_j1_77_60, i_top_j2_77_60, i_top_rej_77_60, i_Zprime_77_60, ev.EventIndex)
    Containers["inc_top_77_0" ].Add(i_top_lep_77_0 , i_top_j1_77_0 , i_top_j2_77_0 , i_top_rej_77_0 , i_Zprime_77_0 , ev.EventIndex)
                                                                                                     
    Containers["inc_top_70_85"].Add(i_top_lep_70_85, i_top_j1_70_85, i_top_j2_70_85, i_top_rej_70_85, i_Zprime_70_85, ev.EventIndex)
    Containers["inc_top_70_77"].Add(i_top_lep_70_77, i_top_j1_70_77, i_top_j2_70_77, i_top_rej_70_77, i_Zprime_70_77, ev.EventIndex)
    Containers["inc_top_70_70"].Add(i_top_lep_70_70, i_top_j1_70_70, i_top_j2_70_70, i_top_rej_70_70, i_Zprime_70_70, ev.EventIndex)
    Containers["inc_top_70_60"].Add(i_top_lep_70_60, i_top_j1_70_60, i_top_j2_70_60, i_top_rej_70_60, i_Zprime_70_60, ev.EventIndex)
    Containers["inc_top_70_0" ].Add(i_top_lep_70_0 , i_top_j1_70_0 , i_top_j2_70_0 , i_top_rej_70_0 , i_Zprime_70_0 , ev.EventIndex)
                                                                                                      
    Containers["inc_top_60_85"].Add(i_top_lep_60_85, i_top_j1_60_85, i_top_j2_60_85, i_top_rej_60_85, i_Zprime_60_85, ev.EventIndex)
    Containers["inc_top_60_77"].Add(i_top_lep_60_77, i_top_j1_60_77, i_top_j2_60_77, i_top_rej_60_77, i_Zprime_60_77, ev.EventIndex)
    Containers["inc_top_60_70"].Add(i_top_lep_60_70, i_top_j1_60_70, i_top_j2_60_70, i_top_rej_60_70, i_Zprime_60_70, ev.EventIndex)
    Containers["inc_top_60_60"].Add(i_top_lep_60_60, i_top_j1_60_60, i_top_j2_60_60, i_top_rej_60_60, i_Zprime_60_60, ev.EventIndex)
    Containers["inc_top_60_0" ].Add(i_top_lep_60_0 , i_top_j1_60_0 , i_top_j2_60_0 , i_top_rej_60_0 , i_Zprime_60_0 , ev.EventIndex)



if __name__ == '__main__':
    Containers = {}

    Containers["Truth"] = Container("Truth")
    Containers["top_85_85"] = Container("85_85")
    Containers["top_85_77"] = Container("85_77")
    Containers["top_85_70"] = Container("85_70")
    Containers["top_85_60"] = Container("85_60")
    Containers["top_85_0" ] = Container("85_0" )

    Containers["top_77_85"] = Container("77_85")
    Containers["top_77_77"] = Container("77_77")
    Containers["top_77_70"] = Container("77_70")
    Containers["top_77_60"] = Container("77_60")
    Containers["top_77_0" ] = Container("77_0" )

    Containers["top_70_85"] = Container("70_85")
    Containers["top_70_77"] = Container("70_77")
    Containers["top_70_70"] = Container("70_70")
    Containers["top_70_60"] = Container("70_60")
    Containers["top_70_0" ] = Container("70_0" )

    Containers["top_60_85"] = Container("60_85")
    Containers["top_60_77"] = Container("60_77")
    Containers["top_60_70"] = Container("60_70")
    Containers["top_60_60"] = Container("60_60")
    Containers["top_60_0" ] = Container("60_0" )


    Containers["inc_top_85_85"] = Container("inc_85_85")
    Containers["inc_top_85_77"] = Container("inc_85_77")
    Containers["inc_top_85_70"] = Container("inc_85_70")
    Containers["inc_top_85_60"] = Container("inc_85_60")
    Containers["inc_top_85_0" ] = Container("inc_85_0" )

    Containers["inc_top_77_85"] = Container("inc_77_85")
    Containers["inc_top_77_77"] = Container("inc_77_77")
    Containers["inc_top_77_70"] = Container("inc_77_70")
    Containers["inc_top_77_60"] = Container("inc_77_60")
    Containers["inc_top_77_0" ] = Container("inc_77_0" )

    Containers["inc_top_70_85"] = Container("inc_70_85")
    Containers["inc_top_70_77"] = Container("inc_70_77")
    Containers["inc_top_70_70"] = Container("inc_70_70")
    Containers["inc_top_70_60"] = Container("inc_70_60")
    Containers["inc_top_70_0" ] = Container("inc_70_0" )

    Containers["inc_top_60_85"] = Container("inc_60_85")
    Containers["inc_top_60_77"] = Container("inc_60_77")
    Containers["inc_top_60_70"] = Container("inc_60_70")
    Containers["inc_top_60_60"] = Container("inc_60_60")
    Containers["inc_top_60_0" ] = Container("inc_60_0" )




    File = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/tttt/" 
    Ana = Analysis()
    Ana.ProjectName = "SingleLepton"
    Ana.InputSample("4Tops", {File : ["*"]})
    Ana.Event = Event
    Ana.EventCache = False
    Ana.DumpPickle = True
    Ana.chnk = 100
    Ana.Launch()
    
    le = len(Ana)
    it = 0
    for i in Ana:
        SingleLeptonAnalysis(Containers, i)
        print(it, le) 
        it += 1 

    Tru = Containers["Truth"]
    Tru.CalculateLumi(Tru)
    for key in Containers:
        Containers[key].MakeAllMass()
        Containers[key].CalculateLumi(Tru)

    PickleObject(Containers, "FullContainers") 



    def PlotMass(Cont, Title, OutDir):
        Plots = {}
        Plots["Style"] = "ATLAS"
        Plots["xBins"] = 500
        Plots["xMax"] = 2000
        Plots["xMin"] = 0
        Plots["xTitle"] = "Mass (GeV)"
        Plots["yTitle"] = "Entries"
        Plots["Title"] = Title
        Plots["Filename"] = Cont.WorkingPoint
        Plots["OutputDirectory"] = OutDir
        Plots["ATLASLumi"] = Cont.Lumi
        TH = CombineTH1F(**Plots)
        
        #P1 = {}
        #P1 |= Plots
        #P1["Title"] = "Reco-Leptonic"
        #P1["xData"] = Cont.lepT
        #P1 = TH1F(**P1)
        #P1.Compile()

        P2 = {}
        P2 |= Plots
        P2["Title"] = "Residual-Jets"
        P2["xData"] = Cont.resT
        P2 = TH1F(**P2)
        P2.Compile()
        
        P3 = {}
        P3 |= Plots
        P3["Title"] = "Reco-Hardest-btag-Jets"
        P3["xData"] = Cont.t1T + Cont.t2T
        P3 = TH1F(**P3)
        P3.Compile()

        P4 = {}
        P4 |= Plots
        P4["Title"] = "Z'-From-Hardest-Jets"
        P4["xData"] = Cont.Zprime
        P4["Filename"] = Cont.WorkingPoint + "_RecoZprime"
        P4 = TH1F(**P4)
        P4.SaveFigure()

        TH.Histograms = [P2, P3]
        TH.SaveFigure()



    PlotMass(Containers["inc_top_85_85"], "Non Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["inc_top_85_77"], "Non Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["inc_top_85_70"], "Non Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["inc_top_85_60"], "Non Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["inc_top_85_0" ], "Non Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    
    PlotMass(Containers["inc_top_77_85"], "Non Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["inc_top_77_77"], "Non Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["inc_top_77_70"], "Non Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["inc_top_77_60"], "Non Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["inc_top_77_0" ], "Non Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (77%) Jets" , "./Plots/wrk77/")
 
    PlotMass(Containers["inc_top_70_85"], "Non Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["inc_top_70_77"], "Non Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["inc_top_70_70"], "Non Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["inc_top_70_60"], "Non Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["inc_top_70_0" ], "Non Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (70%) Jets" , "./Plots/wrk70/")
 
    PlotMass(Containers["inc_top_60_85"], "Non Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["inc_top_60_77"], "Non Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["inc_top_60_70"], "Non Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["inc_top_60_60"], "Non Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["inc_top_60_0" ], "Non Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (60%) Jets" , "./Plots/wrk60/")


    PlotMass(Containers["top_85_85"], "Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["top_85_77"], "Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["top_85_70"], "Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["top_85_60"], "Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    PlotMass(Containers["top_85_0" ], "Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (85%) Jets", "./Plots/wrk85/")
    
    PlotMass(Containers["top_77_85"], "Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["top_77_77"], "Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["top_77_70"], "Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["top_77_60"], "Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (77%) Jets", "./Plots/wrk77/")
    PlotMass(Containers["top_77_0" ], "Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (77%) Jets" , "./Plots/wrk77/")
 
    PlotMass(Containers["top_70_85"], "Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["top_70_77"], "Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["top_70_70"], "Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["top_70_60"], "Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (70%) Jets", "./Plots/wrk70/")
    PlotMass(Containers["top_70_0" ], "Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (70%) Jets" , "./Plots/wrk70/")
 
    PlotMass(Containers["top_60_85"], "Mutually Exclusive Leptonically Matched Jets (85%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["top_60_77"], "Mutually Exclusive Leptonically Matched Jets (77%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["top_60_70"], "Mutually Exclusive Leptonically Matched Jets (70%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["top_60_60"], "Mutually Exclusive Leptonically Matched Jets (60%) \n to two Hardest B-Tagged (60%) Jets", "./Plots/wrk60/")
    PlotMass(Containers["top_60_0" ], "Mutually Exclusive Leptonically Matched Jets (0%)  \n to two Hardest B-Tagged (60%) Jets" , "./Plots/wrk60/")
 
    PlotMass(Containers["Truth" ], "Z' Expected from Truth matched Reconstructed Jets" , "./Plots/Truth/")
 

