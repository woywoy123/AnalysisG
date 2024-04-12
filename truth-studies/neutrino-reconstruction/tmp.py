

import sys
sys.path.append("../../test/neutrino_reconstruction/")
from nusol import (SingleNu, DoubleNu)
from AnalysisG.Templates import SelectionTemplate

class NeutrinoReconstruction(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.num_sols = {"mev" : [], "gev" : []}

        self.top_mass_r2l = {"mev" : [], "gev" : []}
        self.top_mass_r1l = {"mev" : [], "gev" : []}
        self.H_mass_r2l = {"mev" : [], "gev" : []}

        self.top_mass_t1l = {"children" : []}
        self.top_mass_t2l = {"children" : []}
        self.H_mass_t2l = {"children" : []}

        self.top_kin_r2l = {
                "mev" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "gev" : {"px" : [], "py" : [], "pz" : [], "e" : []},
        }

        self.top_kin_r1l = {
                "mev" : {"px" : [], "py" : [], "pz" : [], "e" : []},
                "gev" : {"px" : [], "py" : [], "pz" : [], "e" : []},
        }

    def Selection(self, event):
        self.leps = len([1 for t in event.Tops if t.LeptonicDecay])
        if self.leps > 2 and self.leps > 0: return False
        return True

    def kinCollector(self, dic, key, reco, truth):
        dic[key]["px"].append( abs((truth.px - reco.px)/reco.px)*100 )
        dic[key]["py"].append( abs((truth.py - reco.py)/reco.py)*100 )
        dic[key]["pz"].append( abs((truth.pz - reco.pz)/reco.pz)*100 )
        dic[key]["e" ].append( abs((truth.e  - reco.e )/reco.e )*100 )

    def SingleNeutrino(self, event, t1):
        b1  = [c for c in t1.Children if c.is_b][0]
        l1  = [c for c in t1.Children if c.is_lep][0]
        nu1 = [c for c in t1.Children if c.is_nu][0]

        mT = (b1+l1+nu1).Mass
        mW = (l1 + nu1).Mass
        tvl = [nu+b1+l1 for nu in self.Nu(b1, l1, event, mT = mT, mW = mW)]
        self.num_sols["mev"] += [len(tvl)]
        for t in tvl:
            self.top_mass_r1l["mev"] += [t.Mass/1000]
            self.kinCollector(self.top_kin_r1l, "mev", t, t1)

        tvl = [nu+b1+l1 for nu in self.Nu(b1, l1, event, mT = mT, mW = mW, gev = True)]
        self.num_sols["gev"] += [len(tvl)]
        for t in tvl:
            self.top_mass_r1l["gev"] += [t.Mass/1000]
            self.kinCollector(self.top_kin_r1l, "gev", t, t1)

        self.top_mass_t1l["children"].append((b1+l1+nu1).Mass/1000)
        return

    def DileptonNeutrino(self, event, t1, t2):
        b1  = [c for c in t1.Children if c.is_b][0]
        l1  = [c for c in t1.Children if c.is_lep][0]
        nu1 = [c for c in t1.Children if c.is_nu][0]

        b2  = [c for c in t2.Children if c.is_b][0]
        l2  = [c for c in t2.Children if c.is_lep][0]
        nu2 = [c for c in t2.Children if c.is_nu][0]

        from_res = t1.FromRes == t2.FromRes == 1
        if from_res:
            H = (b1 + b2 + l1 + l2 + nu1 + nu2)
            self.H_mass_t2l["children"].append(H.Mass/1000)
        self.top_mass_t2l["children"] += [(b1+l1+nu1).Mass/1000, (b2+l2+nu2).Mass/1000]

        mT = (b1+l1+nu1).Mass
        mW = (l1 + nu1).Mass

        nus = self.NuNu(b1, b2, l1, l2, event, mT = mT, mW = mW, gev = False)
        self.num_sols["mev"] += [len(nus)]
        for nu_p in nus:
            tvl  = nu_p[0] + b1 + l1
            tvl_ = nu_p[1] + b2 + l2

            self.top_mass_r2l["mev"] += [ tvl.Mass/1000]
            self.top_mass_r2l["mev"] += [tvl_.Mass/1000]
            if from_res: self.H_mass_r2l["mev"] += [(tvl + tvl_).Mass/1000]

            self.kinCollector(self.top_kin_r2l, "mev", tvl , t1)
            self.kinCollector(self.top_kin_r2l, "mev", tvl_, t2)

        nus = self.NuNu(b1, b2, l1, l2, event, mT = mT, mW = mW, gev = True)
        self.num_sols["gev"] += [len(nus)]
        for nu_p in nus:
            tvl  = nu_p[0] + b1 + l1
            tvl_ = nu_p[1] + b2 + l2

            self.top_mass_r2l["gev"] += [ tvl.Mass/1000]
            self.top_mass_r2l["gev"] += [tvl_.Mass/1000]
            if from_res: self.H_mass_r2l["gev"] += [(tvl + tvl_).Mass/1000]

            self.kinCollector(self.top_kin_r2l, "gev", tvl, t1)
            self.kinCollector(self.top_kin_r2l, "gev", tvl_, t2)

    def Strategy(self, event):
        leptops = [t for t in event.Tops if t.LeptonicDecay]
        if self.leps == 2:
            t1, t2 = leptops
            self.DileptonNeutrino(event, t1, t2)
        else:
            self.SingleNeutrino(event, leptops[0])











from AnalysisG.Plotting import TH1F, TH2F
from selection import NeutrinoReconstruction
from AnalysisG.Events import Event
from AnalysisG.IO import nTupler, PickleObject, UnpickleObject
from AnalysisG import Analysis

import os
smpl = os.environ["Samples"]

mass = "900"
#Ana = Analysis()
#Ana.ProjectName = "Project_Nu"
#Ana.OutputDirectory = "./Reconstruction"
#Ana.InputSample("bsm" + mass, smpl+"/ttZ-" + mass + "/")
#Ana.AddSelection(NeutrinoReconstruction)
#Ana.EventCache = True
#Ana.Event = Event
#Ana.Threads = 10
#Ana.Chunks = 1000
#Ana.Launch()

#x = nTupler()
#x.ProjectName = "Project_Nu"
#x.OutputDirectory = "./Reconstruction"
#x.This("NeutrinoReconstruction -> ", "nominal")
#x.Threads = 12
#x.Chunks = 1000
#nu = x.merged()["nominal.NeutrinoReconstruction"]
#PickleObject(nu.__getstate__())

nu = NeutrinoReconstruction()
nu.__setstate__(UnpickleObject())

# Compare the number of solutions for GeV and MeV 
thc = TH1F()
thc.xMin = 0
thc.xStep = 1
thc.xBinCentering = True
thc.Stack = True
thc.Title = "Number of Neutrino Solutions Using MeV vs GeV"
thc.xTitle = "Number of Solutions"
thc.yTitle = "Entries (arb.)"
thc.Filename = "neutrino-solutions"
thc.OutputDirectory = "./Plotting"

for key in nu.num_sols:
    th = TH1F()
    th.Title = "NSols-" + key
    th.xData = nu.num_sols[key]
    thc.Histograms.append(th)
thc.SaveFigure()

# Compare the mass distribution of the reconstructed tops to the truth
thc = TH1F()
thc.xMin = 0
thc.xStep = 20
thc.xMax = 400
thc.yLogarithmic = True
thc.Title = "Reconstructed Top Invariant Mass Distribution From Neutrinos \n (Dilepton Channel)"
thc.xTitle = "Invariant Mass (GeV)"
thc.yTitle = "Entries <unit>"
thc.OutputDirectory = "./Plotting"
thc.Filename = "top-dilepton"

for key in nu.top_mass_r2l:
    th = TH1F()
    th.Title = "reco using " + key
    th.xData = nu.top_mass_r2l[key]
    thc.Histograms.append(th)

th = TH1F()
th.Title = "truth"
th.xData = nu.top_mass_t2l["children"]
thc.Histogram = th
thc.SaveFigure()

# Compare the mass distribution of the reconstructed tops to the truth
thc = TH1F()
thc.xMin = 0
thc.xStep = 20
thc.xMax = 400
thc.yLogarithmic = True
thc.Title = "Reconstructed Top Invariant Mass Distribution From Neutrinos \n (Single Lepton Channel)"
thc.xTitle = "Invariant Mass (GeV)"
thc.yTitle = "Entries <unit>"
thc.OutputDirectory = "./Plotting"
thc.Filename = "top-singlelepton"

for key in nu.top_mass_r1l:
    th = TH1F()
    th.Title = "reco using " + key
    th.xData = nu.top_mass_r1l[key]
    thc.Histograms.append(th)

th = TH1F()
th.Title = "truth"
th.xData = nu.top_mass_t1l["children"]

thc.Histogram = th
thc.SaveFigure()

# Compare the mass distribution of the reconstructed H to the truth
thc = TH1F()
thc.xMin = 0
thc.xBins = 200
thc.xStep = 200
thc.xMax = 2200
thc.yLogarithmic = True
thc.Title = "Reconstructed Scalar H Invariant Mass Distribution \n From Neutrinos (Dilepton Channel)"
thc.yTitle = "Entries <unit>"
thc.xTitle = "Invariant Mass (GeV)"
thc.OutputDirectory = "./Plotting"
thc.Filename = "H-dilepton"

for key in nu.top_mass_r1l:
    th = TH1F()
    th.Title = "reco using " + key
    th.xData = nu.H_mass_r2l[key]
    thc.Histograms.append(th)

th = TH1F()
th.Title = "truth"
th.xData = nu.H_mass_t2l["children"]
thc.Histogram = th

thc.SaveFigure()

# ______ Compare the kinematics between MeV and GeV with truth ___________ #
# ================================== px - py ======================================= 
th2 = TH2F()
th2.Title = "Difference in (x, y)-Momenta between Reconstructed and Truth Tops \n From Neutrinos using MeV (Dilepton Channel)"
th2.xTitle = "$100 \\cdot |\\Delta px_{MeV}|/(px_{truth})$ - (\\%)"
th2.yTitle = "$100 \\cdot |\\Delta py_{MeV}|/(py_{truth})$ - (\\%)"

th2.xBins = 100
th2.yBins = 100

th2.xMin = 0
th2.xMax = 100

th2.yMin = 0
th2.yMax = 100

th2.xOverFlow = True
th2.yOverFlow = True

th2.xData = nu.top_kin_r2l["mev"]["px"]
th2.yData = nu.top_kin_r2l["mev"]["py"]

th2.Filename = "top-double-kin-px_py-mev"
th2.OutputDirectory = "./Plotting/"
th2.SaveFigure()

th2 = TH2F()
th2.Title = "Difference in (x, y)-Momenta between Reconstructed and Truth Tops \n From Neutrinos using GeV (Dilepton Channel)"
th2.xTitle = "$100 x |\\Delta px_{GeV}|/(px_{truth})$ - (\\%)"
th2.yTitle = "$100 x |\\Delta py_{GeV}|/(py_{truth})$ - (\\%)"

th2.xBins = 100
th2.yBins = 100

th2.xMin = 0
th2.xMax = 100

th2.yMin = 0
th2.yMax = 100

th2.xOverFlow = True
th2.yOverFlow = True

th2.xData = nu.top_kin_r2l["gev"]["px"]
th2.yData = nu.top_kin_r2l["gev"]["py"]

th2.Filename = "top-double-kin-px_py-gev"
th2.OutputDirectory = "./Plotting/"
th2.SaveFigure()

# ================================== e - pz ======================================= 
th2 = TH2F()
th2.Title = "Difference in (e, z)-Momenta between Reconstructed and Truth Tops \n From Neutrinos using MeV (Dilepton Channel)"
th2.xTitle = "$100 x |\\Delta pz_{MeV}|/(pz_{truth})$ - (\\%)"
th2.yTitle = "$100 x |\\Delta e_{MeV}|/(e_{truth})$ - (\\%)"

th2.xBins = 100
th2.yBins = 100

th2.xMin = 0
th2.xMax = 100

th2.yMin = 0
th2.yMax = 100

th2.xOverFlow = True
th2.yOverFlow = True

th2.xData = nu.top_kin_r2l["mev"]["pz"]
th2.yData = nu.top_kin_r2l["mev"]["e"]

th2.Filename = "top-double-kin-pz_e-mev"
th2.OutputDirectory = "./Plotting/"
th2.SaveFigure()

th2 = TH2F()
th2.Title = "Difference in (e, z)-Momenta between Reconstructed and Truth Tops \n From Neutrinos using GeV (Dilepton Channel)"
th2.xTitle = "$100 x |\\Delta pz_{GeV}|/(pz_{truth})$ - (\\%)"
th2.yTitle = "$100 x |\\Delta e_{GeV}|/(e_{truth})$ - (\\%)"

th2.xBins = 100
th2.yBins = 100

th2.xMin = 0
th2.xMax = 100

th2.yMin = 0
th2.yMax = 100

th2.xOverFlow = True
th2.yOverFlow = True

th2.xData = nu.top_kin_r2l["gev"]["pz"]
th2.yData = nu.top_kin_r2l["gev"]["e"]

th2.Filename = "top-double-kin-pz_e-gev"
th2.OutputDirectory = "./Plotting/"
th2.SaveFigure()

