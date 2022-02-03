from Functions.IO.IO import File, PickleObject, UnpickleObject

def TestSimpleTruthMatching():
    
    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/output.root"
    F = File(Dir, True)
    Trees = ["nominal"]
    
    F.Leaves += ["truthjet_pt","truthjet_e", "truthjet_eta", "truthjet_phi", "truthjet_pdgid"]
    F.Leaves += ["truth_top_pt", "truth_top_e", "truth_top_eta", "truth_top_phi", "truth_top_FromRes"]
    F.Leaves += ["topPreFSR_pt", "topPreFSR_e", "topPreFSR_eta", "topPreFSR_phi", "topPreFSR_charge"]
    F.Leaves += ["topPostFSR_pt", "topPostFSR_e", "topPostFSR_eta", "topPostFSR_phi", "topPostFSR_charge"]

    F.Leaves += ["topPostFSRchildren_pt", "topPostFSRchildren_e", "topPostFSRchildren_eta", "topPostFSRchildren_phi", "topPostFSRchildren_charge"]
    F.Leaves += ["truth_top_child_pt", "truth_top_child_e", "truth_top_child_eta", "truth_top_child_phi", "truth_top_child_charge", "truth_top_child_pdgid"]
    F.Leaves += ["GhostTruthJetMap"]



    F.Trees += Trees
    F.CheckKeys() 
    F.ConvertToArray()
    print(F.ArrayLeaves)


    print(F)

    return True
