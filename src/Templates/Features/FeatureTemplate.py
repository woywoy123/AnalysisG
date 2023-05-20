import AnalysisG.Templates.Features.TruthTop as T
import AnalysisG.Templates.Features.TruthTopChildren as TC
import AnalysisG.Templates.Features.TruthJet as TJ
import AnalysisG.Templates.Features.Jet

def TruthTops():
    # Edge: Truth 
    ET = {"res_edge" : T.Edge.res_edge}

    # Node: Truth 
    NT = {"res_node" : T.Node.res_node}

    # Graph: Truth 
    GT = {
            "signal" : T.Graph.signal, 
            "ntops" : T.Graph.ntops, 
    }

    # Node: Feature 
    NF = {
            "eta" : T.Node.eta, 
            "energy" : T.Node.energy, 
            "pT" : T.Node.pT, 
            "phi" : T.Node.phi, 
    }
    
    # Graph: Feature 
    GF = {
            "met" : T.Graph.met, 
            "phi" : T.Graph.phi, 
    }

    Features = {}
    Features |= AddFeature("ET", ET)
    Features |= AddFeature("NT", NT)
    Features |= AddFeature("GT", GT)

    Features |= AddFeature("NF", NF)
    Features |= AddFeature("GF", GF)

    return Features

def TruthChildren():

    # Edge: Truth 
    ET = {
            "res_edge" : TC.Edge.res_edge, 
            "top_edge" : TC.Edge.top_edge, 
            "lep_edge" : TC.Edge.lep_edge, 
    }

    # Node: Truth 
    NT = {
            "res_node" : TC.Node.res_node, 
    }

    # Graph: Truth 
    GT = {
            "signal" : TC.Graph.signal, 
            "ntops" : TC.Graph.ntops,
            "n_nu" : TC.Graph.n_nu,  
    }

    # Node: Feature 
    NF = {
            "eta" : TC.Node.eta, 
            "energy" : TC.Node.energy, 
            "pT" : TC.Node.pT, 
            "phi" : TC.Node.phi, 
            "is_b" : TC.Node.is_b, 
            "is_lep" : TC.Node.is_lep, 
            "is_nu" : TC.Node.is_nu, 
    }
 
    # Graph: Feature 
    GF = {
            "met" : TC.Graph.met, 
            "phi" : TC.Graph.phi, 
    }

    Features = {}
    Features |= AddFeature("ET", ET)
    Features |= AddFeature("NT", NT)
    Features |= AddFeature("GT", GT)

    Features |= AddFeature("NF", NF)
    Features |= AddFeature("GF", GF)

    return Features


def AddFeature(Prefix, dic):
    return {Prefix + "_" + i : dic[i] for i in dic} 

def ApplyFeatures(A, Level):
    if Level == "TruthTops": Features = TruthTops()
    elif Level == "TruthChildren": Features = TruthChildren()

    for i in Features:
        base = "_".join(i.split("_")[1:])
        fx = Features[i]
        
        if "EF" in i: A.AddEdgeFeature(fx, base)
        elif "NF" in i: A.AddNodeFeature(fx, base)
        elif "GF" in i: A.AddGraphFeature(fx, base)
        elif "ET" in i: A.AddEdgeTruth(fx, base)
        elif "NT" in i: A.AddNodeTruth(fx, base)
        elif "GT" in i: A.AddGraphTruth(fx, base)

