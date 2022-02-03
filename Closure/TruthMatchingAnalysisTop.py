from Functions.IO.IO import File, PickleObject, UnpickleObject

def TestSimpleTruthMatching():
    
    #Dir = "/home/tnom6927/Downloads/SimpleTTBAR/output.root"
    #F = File(Dir)
    #F.Trees = ["nominal"]
    #kin = ["eventNumber"]
    #el = ["el_pt", "el_eta", "el_phi", "el_e", "el_true_type", "el_true_origin"]
    #mu = ["mu_pt", "mu_eta", "mu_phi", "mu_e", "mu_true_type", "mu_true_origin"]
    #jet = ["jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_truthflav", "jet_truthPartonLabel"]
    #
    #F.Leaves += kin
    #F.Leaves += el
    #F.Leaves += mu
    #F.Leaves += jet

    #F.CheckKeys()
    #F.ConvertToArray()
   
    #T = File(Dir)
    #T.Trees = ["truth"]
    #kin = ["eventNumber"]
    #mc = ["mc_pt", "mc_eta", "mc_phi", "mc_e", "mc_pdgId"]
    #kine = ["_pdgId", "_eta", "_phi", "_pt"]
    #MC_ = ["MC_Wdecay1_from_tbar", "MC_Wdecay1_from_t", "MC_Wdecay2_from_tbar", "MC_Wdecay2_from_t", "MC_b_from_t", "MC_W_from_t", "MC_b_from_tbar", "MC_W_from_tbar"]
    #
    #new = []
    #for i in MC_:
    #    for k in kine:
    #        new.append(i + k)

    #T.Leaves += kin
    #T.Leaves += mc
    #T.Leaves += new

    #T.CheckKeys()
    #T.ConvertToArray()

    #F = [F, T]
    #PickleObject(F, "TruthMatching.pkl")
    F = UnpickleObject("TruthMatching.pkl") 

    T = F[1].ArrayLeaves
    F = F[0].ArrayLeaves
    x = 0
    for l in range(len(F["nominal/eventNumber"])):
        k = F["nominal/eventNumber"][l]
        for t in range(x, len(T["truth/eventNumber"])):
            if k == T["truth/eventNumber"][t]:
                for i in T["truth/mc_pdgId"][t]:
                    print(i)
                print(k, F["nominal/jet_truthflav"][l])

                x += 1
                break

        break


    return True
