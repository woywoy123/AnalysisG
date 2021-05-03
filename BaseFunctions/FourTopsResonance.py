from BaseFunctions.IO import *
from BaseFunctions.UpRootFunctions import *
import ROOT

def ReadLeafsFromResonance(file_dir):

    Entry_Objects = ObjectsFromFile(file_dir, "nominal", ["top_FromRes", "truth_top_pt", "truth_top_eta", "truth_top_phi", "truth_top_e"])
    skimmed = FileObjectsToArrays(Entry_Objects)
    branch_map = skimmed["nominal"]

    Simple_TopMass = []
    Resonance_Mass = []

    for i in range(len(branch_map["top_FromRes"])):
        res_v = branch_map["top_FromRes"][i]
        pt_v = branch_map["truth_top_pt"][i]
        eta_v = branch_map["truth_top_eta"][i]
        phi_v = branch_map["truth_top_phi"][i]
        e_v = branch_map["truth_top_e"][i]
       
        lor_pair = []
        # Create a vector element for each top
        for x in range(len(res_v)):
            v = ROOT.Math.PtEtaPhiEVector()

            if res_v[x] != 0:
                lor_pair.append(v.SetCoordinates(pt_v[x], eta_v[x], phi_v[x], e_v[x]))
            else: 
                Simple_TopMass.append(v.SetCoordinates(pt_v[x], eta_v[x], phi_v[x], e_v[x]).mass())

            # Collect the individual top masses 
            Simple_TopMass.append(float(v.mass()) / 1000)

        # Get the mass of the resonance   
        if len(lor_pair) != 2:
            continue
        delta = lor_pair[0] + lor_pair[1]
        Resonance_Mass.append(float(delta.mass()) / 1000)
       
    return Simple_TopMass, Resonance_Mass

