from AnalysisG.Templates import SelectionTemplate

class SingleLepton(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.ZMass = {"85" : [], "77" : [], "70" : [], "60" : [], "Truth" : []}

    def Strategy(self, event):
        jets = event.Jets
        tops = event.Tops
        
        hardest = {j.pt/1000 : j for j in jets}
        l = list(hardest)
        l.sort(reverse = True)
        modes_d = {}
        hard85_j = [hardest[j] for j in l if hardest[j].btag_DL1r_85 == 1]
        hard77_j = [hardest[j] for j in l if hardest[j].btag_DL1r_77 == 1]
        hard70_j = [hardest[j] for j in l if hardest[j].btag_DL1r_70 == 1]
        hard60_j = [hardest[j] for j in l if hardest[j].btag_DL1r_60 == 1]
        modes_d["85"] = hard85_j
        modes_d["77"] = hard77_j
        modes_d["70"] = hard70_j
        modes_d["60"] = hard60_j

        mode = ""
        if len(hard85_j) >= 2:   mode = "85"
        elif len(hard85_j) >= 2: mode = "85"        
        elif len(hard77_j) >= 2: mode = "77"
        elif len(hard70_j) >= 2: mode = "70"
        elif len(hard60_j) >= 2: mode = "60"
        else: return "Failed -> Not Enough b-tagged"
        
        j1, j2 = modes_d[mode][:2]
        dR = {}
        for j_ in jets:
            if j_ in [j1, j2]: continue
            dR[j1.DeltaR(j_)] = [j1, j_] 
            dR[j2.DeltaR(j_)] = [j2, j_]             

        lst = [j1, j2]
        indx = {0 : [j1], 1 : [j2]}
        dr = list(dR)
        dr.sort()
        for r in dr:
            j, j_ = dR[r]
            ith = lst.index(j)
            if len(indx[ith]) > 2: continue
            indx[ith].append(j_)
          
        if len(indx[0] + indx[1]) != 6: return "Failed -> Not Enough Jets to Match"
        self.ZMass[mode].append(sum(indx[0] + indx[1]).Mass/1000)
        
        tops = [t for t in tops if t.FromRes == 1]
        if len(tops) == 0: return "Failed -> No Resonance Tops"
        t_objects = []
        for t in tops:
            if t.LeptonicDecay: t_objects += [c for c in t.Children if c.is_nu or c.is_lep]
            t_objects += t.Jets    
        self.ZMass["Truth"].append(sum(t_objects).Mass / 1000)                 
     
