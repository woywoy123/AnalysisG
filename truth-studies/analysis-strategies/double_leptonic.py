from AnalysisG.Templates import SelectionTemplate

class DiLeptonic(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.Masses = {
                "Lep-Had" : [], 
                "Lep-Lep" : [], 
                "Had-Had" : []
        }
        self.Process = ""
        self.AllowFailure = False
        self._truthmode = "children"

    def ParticleRouter(self, event):
        if self._truthmode == "children": 
            particles = [i for i in event.TopChildren]
        return particles


    def Selection(self, event):
        particles = self.ParticleRouter(event)
        if len([i for i in particles if i.is_lep]) != 2: return False
        if len([i for i in particles if i.is_b]) != 4: return False
        return True

    def Highest_b(self, particles):
        if self._truthmode == "children":
            highb = { c.pt : c for c in particles if c.is_b}
        highpt = sorted(highb, reverse = True)
        if len(highb) >= 2: highpt = highpt[:2]
        high_ptb = [highb[c] for c in highpt]
        return high_ptb

    def Strategy(self, event):
        particles = self.ParticleRouter(event)

        # find the highest pT b-quarks or bjets
        highb = self.Highest_b(particles)

        # Remove these b's from the list particle candidates and other b quarks
        particles = [ i for i in particles if i not in highb and not i.is_b]

        # Compute the DeltaR between b's
        delR_P = {p.DeltaR(c) : [p, c] for p in highb for c in particles}
        delR = sorted(delR_P)

        # Construct the seeds by selecting only the closest particle from the list 
        # and determining whether the closest was a lepton or quark
        candidates = {x : [x] for x in highb}
        for can in highb:
            it = iter(delR)
            this = None
            dR = None
            while True:
                try: dR = next(it)
                except StopIteration: break

                # check if the candidate is in the deltaR pairs
                if can not in delR_P[dR]: continue

                # Select this pair and make it public
                this = delR_P[dR]

                # Remove this pair from the dic
                delR_P = {r : delR_P[r] for r in delR_P if r != dR}
                delR   = [r for r in delR if r != dR]
                break

            if dR is None: break
            if this is None: continue
            candidates[can].append(this[-1])

        # Now check whether the closest particle is a lepton or quark type
        type_dic = {}
        for can in candidates:
            type_can = candidates[can][-1]
            if can is type_can: type_dic[can] = []
            if type_can.is_lep: type_dic[can] = [can, type_can]
            else:
                # find an additional quark or jet
                type_dic[can] = [can, type_can]
                for dr in delR_P:
                    this = delR_P[dr]
                    if can not in this: continue
                    if this[-1].is_lep: continue
                    type_dic[can].append(this[-1])
                    break
        # if the number of the number of b's is less than 2, reject the event
        if len(type_dic) < 2: return
        if sum([len(type_dic[can]) == 0 for can in type_dic]) == 1: return

        # count the number of leptons in this candidate list
        n_lep = sum([sum([c.is_lep for c in type_dic[can]]) for can in type_dic])
        if n_lep == 0:
            Mass = sum([c for can in type_dic for c in type_dic[can]]).Mass
        elif n_lep == 1:
            Mass = [] 


        print(candidates)
        




        exit()









"""
# Deprecated.....
class MakeTruth(SelectionTemplates):

    def __init__(self):
        SelectionTemplates.__init__(self)

        self.Truth = {
                "Lep" : {"Top" : [], "Children" : [], "TruthJets" : []},
                "Had" : {"Top" : [], "Children" : [], "TruthJets" : []},
             "HadHad" : {"Top" : [], "Children" : [], "TruthJets" : []},
             "LepLep" : {"Top" : [], "Children" : [], "TruthJets" : []},
             "HadLep" : {"Top" : [], "Children" : [], "TruthJets" : []},
             "LepHad" : {"Top" : [], "Children" : [], "TruthJets" : []},
        }
        self.ROOTSamples = []

    def Selection(self, event):

        # Construct All the Truth
        # -> Tops 
        tops = [k for k in event.Tops if len(k.Children) > 0]
        l  = [t for t in tops if t.LeptonicDecay]
        l_ = [t for t in tops if t.LeptonicDecay and t.FromRes == 1]

        h  = [t for t in tops if not t.LeptonicDecay]
        h_ = [t for t in tops if not t.LeptonicDecay and t.FromRes == 1]

        if len(sum(h_ + l_).Children) == 0:
            return "Failed->NoResonanceChildren"

        self.Truth["Lep"]["Top"] += [x.Mass for x in l]
        self.Truth["Had"]["Top"] += [x.Mass for x in h]
        self.Truth["Had"*len(h_) + "Lep"*len(l_)]["Top"] += [sum(h_ + l_).Mass]

        # -> Children 
        self.Truth["Lep"]["Children"] += [ sum(x.Children).Mass for x in l ]
        self.Truth["Had"]["Children"] += [ sum(x.Children).Mass for x in h ]
        self.Truth["Had"*len(h_) + "Lep"*len(l_)]["Children"] += [sum([i for k in h_ + l_ for i in k.Children]).Mass]


class ChildrenSelection(Selection):

    def __init__(self):
        AnalysisTemplate.__init__(self)
        self.Reconstruction = {
               "Lep" : {"Children" : []},
               "Had" : {"Children" : []},
            "HadHad" : {"Children" : []},
            "LepLep" : {"Children" : []},
            "HadLep" : {"Children" : []},
            "LepHad" : {"Children" : []},
        }

        self.lepNum = [11, 13, 15]
        self.nuNumber = [12, 14, 16]
        self.Excl = self.lepNum + self.nuNumber

    def Selection(self, event):
        # Selection 1: on the Children: 
        # - Select the two most energetic children, provided they are quarks.
        # - Assign either leptons or other children to these two based on the lowest deltaR. 
        #   -> If quarks are lowest deltaR: Use two additional quark children 
        #   -> If lepton is matched: Use single/double neutrino reconstruction code.
        # - Derive the Top and the resonance mass
        children = event.TopChildren

        # ---- Get two highest PT quark children ----- # 
        ch = self.Sort({ c.pt/1000 : c for c in children if abs(c.pdgid) not in self.Excl }, True)
        ch = list(ch.values())[:2]
        
        # ----- Remove the two selected children and neutrinos ----- #
        resid_c = [ c for c in children if c not in ch and abs(c.pdgid) not in self.nuNumber ]
        dr = self.Sort({ k.DeltaR(p) : [k, p] for k in resid_c for p in ch})
        
        tmp = []
        cp1, cp2 = [], []
        tl1, tl2 = False, False
        for c in dr:
            p = dr[c][0]
            p_ = dr[c][1]
            
            if p in tmp:
                continue 
            tmp.append(p)
            
            lp = abs(p.pdgid) in self.Excl
            if p_ == ch[0]:
                tl1 = True if lp and len(cp1) == 0 else tl1 
                cp1 += [p] if tl1 and len(cp1) == 0 else []
                cp1 += [p] if not tl1 and len(cp1) < 2 and not lp else []
            if p_ == ch[1]:
                tl2 = True if lp and len(cp2) == 0 else tl2 
                cp2 += [p] if tl2 and len(cp2) == 0 else []
                cp2 += [p] if not tl2 and len(cp2) < 2 and not lp else []
        
        cp1 += [ch[0]]
        cp2 += [ch[1]]
        
        # ------ double Neutrino ----- #
        tl1 = cp1[0].pdgid in self.lepNum
        tl2 = cp2[0].pdgid in self.lepNum
        if len(cp1) == 2 and len(cp2) == 2 and tl1 and tl2:
            o = self.NuNu(cp1[1], cp2[1], cp1[0], cp2[0], event)
            if len(o) == 0:
                return "Failed->NoNuNu"
            m1 = { sum(cp1 + [x[0]]).Mass : cp1 + [x[0]] for x in o }
            m2 = { sum(cp2 + [x[1]]).Mass : cp2 + [x[1]] for x in o }
            o = list(self.Sort({ abs(x1 - x2) : [m1[x1], m2[x2]] for x1, x2 in zip(m1, m2) }).values())[0]
            self.Reconstruction["Lep"]["Children"] += [sum(i).Mass for i in o]
            self.Reconstruction["LepLep"]["Children"] += [sum([i for t in o for i in t]).Mass]
            return "Success->NuNu"
            
        # ------ Single Neutrino ----- #
        if len(cp1) == 2 and tl1:
            o = self.Nu(cp1[1], cp1[0], event)
            if len(o) == 0:
                return "Failed->NoNu"
            t1 = sum(o + cp1)
            t2 = sum(cp2)
            self.Reconstruction["Lep"]["Children"] += [t1.Mass]
            self.Reconstruction["Had"]["Children"] += [t2.Mass]
            self.Reconstruction["LepHad"]["Children"] += [(t1+t2).Mass]
            return "Success->Nu"
            
        # ------ Single Neutrino ----- #
        if len(cp2) == 2 and tl2:
            o = self.Nu(cp2[1], cp2[0], event)
            if len(o) == 0:
                return "Failed->NoNu"
            t2 = sum(o + cp2)
            t1 = sum(cp1)
            self.Reconstruction["Lep"]["Children"] += [t2.Mass]
            self.Reconstruction["Had"]["Children"] += [t1.Mass]
            self.Reconstruction["HadLep"]["Children"] += [(t1+t2).Mass]
            return "Success->Nu"
                
        # ------ All Hadronic ----- #
        t1 = sum(cp1)
        t2 = sum(cp2)
        self.Reconstruction["Had"]["Children"] += [t2.Mass]
        self.Reconstruction["Had"]["Children"] += [t1.Mass]
        self.Reconstruction["HadHad"]["Children"] += [(t1+t2).Mass]
        return "Success->Hadron"
"""
