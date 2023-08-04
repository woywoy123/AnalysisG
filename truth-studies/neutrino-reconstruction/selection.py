from AnalysisG.Templates import SelectionTemplate
import sys
sys.path.append("../../test/neutrino_reconstruction/")
from nusol import (SingleNu, DoubleNu)

class NeutrinoReconstruction(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.num_sols = {"mev" : [], "gev" : []}





    def Selection(self, event):
        self.leps = len([1 for t in event.Tops if t.LeptonicDecay])
        if self.leps > 2 and self.leps > 0: return False
        return True

    def Strategy(self, event):
        leptops = [t for t in event.Tops if t.LeptonicDecay]

        if self.leps == 2: t1, t2 = leptops
        else: t1, t2 = leptops[0], None

        b1  = [c for c in t1.Children if c.is_b][0]
        l1  = [c for c in t1.Children if c.is_lep][0]
        nu1 = [c for c in t1.Children if c.is_nu][0]
        if t2 is None:
            print(self.Nu(b1, l1, event))
            return

        b2  = [c for c in t2.Children if c.is_b][0]
        l2  = [c for c in t2.Children if c.is_lep][0]
        nu2 = [c for c in t2.Children if c.is_nu][0]
        print(self.NuNu(b1, b2, l1, l2, event))

        exit()
