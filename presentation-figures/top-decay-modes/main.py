from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event 
from AnalysisTopGNN.Templates.Selection import Selection 
from AnalysisTopGNN.IO import UnpickleObject
from AnalysisTopGNN.Plotting import TH1F, CombineTH1F 
import sys
import vector
sys.path.append("../../truth-studies/NeutrinoReconstruction/neutrino_momentum_reconstruction.py")

class TopCounter(Selection):

    def __init__(self):
        Selection.__init__(self)
        self.DecayMode = {"Had" : 0, "Lep" : 0, "nTop" : 0}
        self.nLepton = {"None" : 0, "1L" : 0, "2L" : 0, "3L" : 0, "4L" : 0, "" : 0, "nEvents" : 0}

    def Strategy(self, event):
        tops = event.Tops
        counter = 0
        for t in tops:
            mode = "Had" if len([c for c in t.Children if abs(c.pdgid) in [11, 13, 15]]) == 0 else "Lep"
            self.DecayMode[mode] += 1
            if mode == "Lep":
                counter += 1
        self.DecayMode["nTop"] += len(tops)
       
        case = "None" if counter == 0 else ""
        case = "1L" if counter == 1 else case
        case = "2L" if counter == 2 else case
        case = "3L" if counter == 3 else case
        case = "4L" if counter == 4 else case
        self.nLepton[case] += 1
        self.nLepton["nEvents"] += 1

class Truth(Selection):
    def __init__(self):
        Selection.__init__(self)
        self.Tops = {"Had1" : [], "Had2" : [], "Lep1" : [], "Lep2" : []}
        self.Neutrinos = {"Lep1" : [], "Lep2" : []}
        self.Leptons = {"Lep1" : [], "Lep2" : []}
        
        self.Event = {"METx" : [], "METy" : []}
        self.MassTopsTC = {"Had1" : [], "Had2" : [], "Lep1" : [], "Lep2" : []}
        self.MassTopsTJ = {"Had1" : [], "Had2" : [], "Lep1" : [], "Lep2" : []}
        self.MassTopsRECO = {"Had1" : [], "Had2" : [], "Lep1" : [], "Lep2" : []}

        self.EventStats = {"Events" : 0, "Tops" : 0, "True2L" : 0}

    def Selection(self, event):
        tops = event.Tops
        counter = 0
        for t in tops:
            counter += len([c for c in t.Children if abs(c.pdgid) in [11, 13, 15]])
        if counter != 2 or len(tops) != 4:
            return False
        
        self.EventStats["True2L"] += 1
        h, l = 0, 0
        for t in tops:
            if len([c for c in t.Children if abs(c.pdgid) in [11, 13, 15]]) > 0:
                l += 1
                self.Tops["Lep" + str(l)].append(t)
                self.Neutrinos["Lep" + str(l)] += [c for c in t.Children if abs(c.pdgid) in [12, 14, 16]]
                self.Leptons["Lep" + str(l)] += [c for c in t.Children if abs(c.pdgid) in [11, 13, 15]]
            else:
                h += 1
                self.Tops["Had" + str(h)].append(t)
        return True 
            
    def PurgeLastEntry(self):
        for i in self.Tops:
            self.Tops[i] = self.Tops[i][:-1]
        self.Event["METx"] = self.Event["METx"][:-1]
        self.Event["METy"] = self.Event["METy"][:-1]
        for i in self.Neutrinos:
            self.Neutrinos[i] = self.Neutrinos[i][:-1]
            self.Leptons[i] = self.Leptons[i][:-1]
        
        for i in self.MassTopsTJ:
            self.MassTopsTJ[i] = self.MassTopsTJ[i][:-1]
            self.MassTopsRECO[i] = self.MassTopsRECO[i][:-1]


    def Strategy(self, event):
        import PyC.Transform.Floats as TF
        
        tops = event.Tops 
        met = event.met
        phi = event.met_phi
        px, py = TF.Px(met, phi), TF.Py(met, phi)
    
        self.Event["METx"].append(px)
        self.Event["METy"].append(py)
        
        for i in self.MassTopsTC:
            self.MassTopsTC[i] += [sum(self.Tops[i][-1].Children).Mass]
        
        remove = False
        for i in self.Tops:
            # Reject if the top has no truth jets matched 
            if len(self.Tops[i][-1].TruthJets) == 0:
                remove = True
            # Reject if the truth jets have more than one top contributions
            for tj in self.Tops[i][-1].TruthJets:
                if len(tj.Tops) != 1:
                    remove = True

            x = [l for l in self.Tops[i][-1].TruthJets] if remove == False else []
            if "Lep" in i and len(x) > 0:
                nu = self.Neutrinos[i][-1]
                lep = self.Leptons[i][-1]
                x.append(nu) 
                x.append(lep)
                x = [sum(x).Mass]

            elif len(x) > 0:
                x = [sum(x).Mass]
            else:
                x = [0]
            self.MassTopsTJ[i] += x

            # Reject if the top has no jets matched 
            if len(self.Tops[i][-1].Jets) == 0:
                remove = True
            # Reject if the truth jets have more than one top contributions
            for j in self.Tops[i][-1].Jets:
                if len(j.Tops) != 1:
                    remove = True

            x = [l for l in self.Tops[i][-1].Jets] if remove == False else []
            if "Lep" in i and len(x) > 0:
                nu = self.Neutrinos[i][-1] 
                lep = self.Leptons[i][-1]
                x.append(nu) 
                x.append(lep)
                x = [sum(x).Mass] 

            elif len(x) > 0:
                x = [sum(x).Mass]
            else:
                x = [0]

            self.MassTopsRECO[i] += x

        if remove:
            self.PurgeLastEntry()
            return "FAILED->"
        self.EventStats["Events"] += 1
        self.EventStats["Tops"] += len(tops)


def PlotTemplate():
    plt = {}
    plt["xMin"] = 100
    plt["xMax"] = 250
    plt["xStep"] = 1
    plt["xTitle"] = "Invariant Mass (GeV)"
    plt["yTitle"] = "Entries (arb.)"
    return plt

Ana = Analysis()
#Ana.InputSample("bsm-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955717._000001.root")
#Ana.InputSample("bsm-1000-all", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000")
Ana.AddSelection("top-count", TopCounter)
Ana.AddSelection("truth", Truth)
Ana.MergeSelection("top-count")
Ana.MergeSelection("truth")
Ana.Event = Event 
Ana.EventCache = True 
Ana.DumpPickle = True 
Ana.Launch()

#x = Truth()
#x(Ana)

x = UnpickleObject("./UNTITLED/Selections/Merged/truth.pkl")
hsts = []
for i in x.MassTopsTC:
    p = PlotTemplate()
    p["xData"] = x.MassTopsTC[i]
    p["Title"] = i
    hsts.append(TH1F(**p))

p = PlotTemplate()
p["Histograms"] = hsts
p["Title"] = "Top Mass Derived From Truth Children for Dilepton Events"
p["xStep"] = 10
p["Stack"] = True
y = CombineTH1F(**p)
y.Filename = "MassTopsChildren"
y.SaveFigure()

hsts = []
for i in x.MassTopsTJ:
    p = PlotTemplate()
    p["xData"] = x.MassTopsTJ[i]
    p["Title"] = i
    hsts.append(TH1F(**p))

p = PlotTemplate()
p["Histograms"] = hsts
p["Title"] = "Top Mass Derived From Truth Jets (with Truth Neutrino and Lepton if Leptonic)"
p["xStep"] = 10
p["Stack"] = True
y = CombineTH1F(**p)
y.Filename = "MassTopsTruthJets"
y.SaveFigure()

hsts = []
for i in x.MassTopsRECO:
    p = PlotTemplate()
    p["xData"] = x.MassTopsRECO[i]
    p["Title"] = i
    hsts.append(TH1F(**p))

p = PlotTemplate()
p["Histograms"] = hsts
p["Title"] = "Top Mass Derived From Reco Jets (with Truth Neutrino and Lepton if Leptonic)"
p["xStep"] = 10
p["Stack"] = True
y = CombineTH1F(**p)
y.Filename = "MassTopsJets"
y.SaveFigure()

print(x.EventStats)


t = UnpickleObject("./UNTITLED/Selections/Merged/top-count.pkl")
nEvents = t.nLepton["nEvents"]
L0 = (t.nLepton["None"] / nEvents) * 100
L1 = (t.nLepton["1L"] / nEvents) * 100
L2 = (t.nLepton["2L"] / nEvents) * 100
L3 = (t.nLepton["3L"] / nEvents) * 100
L4 = (t.nLepton["4L"] / nEvents) * 100

ntops = t.DecayMode["nTop"]
Had = (t.DecayMode["Had"] / ntops) * 100
Lep = (t.DecayMode["Lep"] / ntops) * 100

print("Decay Mode (%): (HAD)", Had, " (LEP) ", Lep)
print("Lepton Multiplicity (%): (0L)", L0, " (1L)", L1, " (2L)", L2, " (3L)", L3, " (4L)", L4)



