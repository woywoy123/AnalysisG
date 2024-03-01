from AnalysisG.Templates import SelectionTemplate
from itertools import combinations

class AddOnStudies(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.deltaR_lepquark = {"all" : [], "CT" : [], "FT" : []}

        self.nunu_data = {
                "num_sols" : [],
                "had_res" : [], "had_spec" : [], "lep_res" : [], "lep_spec" : [],
                "res_mass" : []
        }

        self.eff_closestLeptonicGroup   = []
        self.eff_remainingLeptonicGroup = []
        self.eff_bestHadronicGroup      = []
        self.eff_remainingHadronicGroup = []
        self.eff_resonance_had          = []
        self.eff_resonance_lep          = []
        self.eff_resonance              = []

    def Selection(self, event): return True

    # Group particles into hadronic groups based on invariant mass and partial leptonic groups based on dR
    def ParticleGroups(self, event):
        lquarks = []
        bquarks = []
        leptons = []
        mT = 172.5*1000  # MeV : t Quark Mass
        for p in event.TopChildren:
            if abs(p.pdgid) < 5: lquarks.append(p)
            elif abs(p.pdgid) == 5: bquarks.append(p)
            elif abs(p.pdgid) in [11, 13, 15]: leptons.append(p)

        # Only keep same-sign dilepton events with 4 b's and 4 non-b's
        l  = len(leptons) != 2
        b  = len(bquarks) != 4
        lq = len(lquarks) != 4
        if l: return 0
        if leptons[0].charge != leptons[1].charge or l or b or lq: return 0

        # Find the group of one b quark and two jets for which the invariant mass is closest to that of a top quark
        groups = {"hadronic": [], "leptonic": []}
        while len(bquarks) > 2:
            lowestError = 1e100
            for b in bquarks:
                for pair in combinations(lquarks, 2):
                    IM = sum([b, pair[0], pair[1]]).Mass
                    if abs(mT - IM) < lowestError:
                        bestB = b
                        bestQuarkPair = pair
                        lowestError = abs(mT - IM)
            # Remove last closest group from lists and add them to dictionary
            bquarks.remove(bestB)
            lquarks.remove(bestQuarkPair[0])
            lquarks.remove(bestQuarkPair[1])
            groups["hadronic"].append([bestB, bestQuarkPair[0], bestQuarkPair[1]])

        # Match remaining leptons with their closest b quarks
        closestPairs = []
        while len(leptons):
            lowestDR = 100
            for l in leptons:
                for b in bquarks:
                    if l.DeltaR(b) < lowestDR:
                        lowestDR = l.DeltaR(b)
                        closestB = b
                        closestL = l
            # Remove last closest lepton and b from lists and add them to dictionary
            leptons.remove(closestL)
            bquarks.remove(closestB)
            groups["leptonic"].append([closestB, closestL])
        return groups


    def Difference(self, leptonic_groups, neutrinos):
        diff = 0
        mT = 172.5*1000  # MeV : t Quark Mass
        for g, group in enumerate(leptonic_groups):
            top_group = sum([group[0], group[1], neutrinos[g]])
            diff += abs(mT - top_group.Mass)
        return diff

    def Strategy(self, event):

        for t1 in event.Tops:
            lep_t = [c for c in t1.Children if c.is_lep]
            if not len(lep_t): continue
            for pair in [(b, t) for t in event.Tops for b in t.Children if b.is_b]:
                b, t2 = pair
                dr = lep_t[0].DeltaR(b)
                self.deltaR_lepquark["all"] += [dr]
                if t1 == t2: self.deltaR_lepquark["CT"].append(dr)
                else: self.deltaR_lepquark["FT"].append(dr)

        grp = self.ParticleGroups(event)
        if not grp: return "Group_Formation_Failed"
        grp["tops"] = {t.index : t for t in event.Tops}

        b_had1 = grp["hadronic"][0][0]
        b_had2 = grp["hadronic"][1][0]

        lep1 = grp["leptonic"][0][1]
        lep2 = grp["leptonic"][1][1]

        nus = self.NuNu(b_had1, b_had2, lep1, lep2, event)
        if not len(nus): return "No_NuNu_Solutions"

        self.nunu_data["num_sols"] += [len(nus)]
        close_T = {self.Difference(grp["leptonic"], nu) : nu for nu in nus}
        x = list(close_T)
        x.sort()
        closest_nu = close_T[x[0]]

        # Make reconstructed tops and assign them to resonance/spectator
        leptonicGroups = [sum([grp["leptonic"][g][0], grp["leptonic"][g][1], closest_nu[g]]) for g in range(2)]
        if leptonicGroups[0].pt > leptonicGroups[1].pt:
            LeptonicResTop = leptonicGroups[0]
            LeptonicSpecTop = leptonicGroups[1]
            nFromRes_leptonicGroup = len([p for p in grp["leptonic"][0] if grp["tops"][p.TopIndex].FromRes == 1])
        else:
            LeptonicResTop = leptonicGroups[1]
            LeptonicSpecTop = leptonicGroups[0]
            nFromRes_leptonicGroup = len([p for p in grp["leptonic"][1] if grp["tops"][p.TopIndex].FromRes == 1])

        hadronicGroups = [sum(grp["hadronic"][g]) for g in range(2)]
        if hadronicGroups[0].pt > hadronicGroups[1].pt:
            HadronicResTop = hadronicGroups[0]
            HadronicSpecTop = hadronicGroups[1]
            nFromRes_hadronicGroup = len([p for p in grp["hadronic"][0] if grp["tops"][p.TopIndex].FromRes == 1])
        else:
            HadronicResTop = hadronicGroups[1]
            HadronicSpecTop = hadronicGroups[0]
            nFromRes_hadronicGroup = len([p for p in grp["hadronic"][1] if grp["tops"][p.TopIndex].FromRes == 1])

        if nFromRes_leptonicGroup == 2: self.eff_resonance_lep += [1]
        if nFromRes_hadronicGroup == 3: self.eff_resonance_had += [1]
        if nFromRes_leptonicGroup == 2 and nFromRes_hadronicGroup == 3: self.eff_resonance += [1]

        # Calculate efficiencies of groupings
        if grp["leptonic"][0][0].TopIndex == grp["leptonic"][0][1].TopIndex: self.eff_closestLeptonicGroup += [1]
        if grp["leptonic"][1][0].TopIndex == grp["leptonic"][1][1].TopIndex: self.eff_remainingLeptonicGroup += [1]

        if grp["hadronic"][0][0].TopIndex == grp["hadronic"][0][1].TopIndex and grp["hadronic"][0][1].TopIndex == grp["hadronic"][0][2].TopIndex:
            self.eff_bestHadronicGroup += [1]

        if grp["hadronic"][1][0].TopIndex == grp["hadronic"][1][1].TopIndex and grp["hadronic"][1][1].TopIndex == grp["hadronic"][1][2].TopIndex:
            self.eff_remainingHadronicGroup += [1]

        # Calculate masses of tops and resonance
        self.nunu_data["had_res"].append(HadronicResTop.Mass/1000)
        self.nunu_data["had_spec"].append(HadronicSpecTop.Mass/1000)
        self.nunu_data["lep_res"].append(LeptonicResTop.Mass/1000)
        self.nunu_data["lep_spec"].append(LeptonicSpecTop.Mass/1000)
        self.nunu_data["res_mass"].append(sum([HadronicResTop, LeptonicResTop]).Mass/1000)

