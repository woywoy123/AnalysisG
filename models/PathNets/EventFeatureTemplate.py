import FeatureTemplates.Generic.EdgeFeature as ef
import FeatureTemplates.Generic.NodeFeature as nf
import FeatureTemplates.Generic.GraphFeature as gf

import FeatureTemplates.TruthTopChildren.NodeFeature as tc_nf

import FeatureTemplates.TruthJet.EdgeFeature as tj_ef
import FeatureTemplates.TruthJet.NodeFeature as tj_nf
import FeatureTemplates.TruthJet.GraphFeature as tj_gf
from AnalysisTopGNN.Tools.ModelTesting import AddFeature

def TruthJets():
    # Node: Generic Particle Properties
    GenPartNF = {
            "eta" : nf.eta, 
            "energy" : nf.energy, 
            "pT" : nf.pT, 
            "phi" : nf.phi, 
            "mass" : nf.mass, 
            "islep" : nf.islepton, 
            "charge" : nf.charge, 
    }
    
    # Graph: Generic Particle Properties
    GenPartGF = {
            "mu" : gf.mu, 
            "met" : gf.met, 
            "met_phi" : gf.met_phi, 
            "pileup" : gf.pileup, 
            "njets" : gf.nTruthJets, 
            "nlep" : gf.nLeptons,
    }
    
    # Truth Edge: Truth Jet Properties
    TruthJetTEF = {
            "edge" : tj_ef.Index, 
    } 
    
    # Truth Node: Truth Jet Properties
    TruthJetTNF = {
            "tops_merged" : tj_nf.TopsMerged, 
            "from_top" : tj_nf.FromTop, 
    } 
    
    # Truth Node: Generic Paritcle Properties
    GenPartTNF = {
            "from_res" : nf.FromRes
    }
    
    # Truth Graph: Generic Paritcle Properties
    GenPartTGF = {
            "mu_actual" : gf.mu_actual,
            "nTops" : gf.nTops, 
            "signal_sample" : gf.SignalSample
    }
    
    Features = {}
    Features |= AddFeature("NF", GenPartNF)
    Features |= AddFeature("GF", GenPartGF)
    Features |= AddFeature("ET", TruthJetTEF)
    Features |= AddFeature("NT", TruthJetTNF)    
    Features |= AddFeature("NT", GenPartTNF)
    Features |= AddFeature("GT", GenPartTGF)
    return Features

def ApplyFeatures(A):
    Features = TruthJets()
    for i in Features:
        base = "_".join(i.split("_")[1:])
        fx = Features[i]
        
        if "EF" in i:
            A.AddEdgeFeature(base, fx)
        elif "NF" in i:
            A.AddNodeFeature(base, fx)
        elif "GF" in i:
            A.AddGraphFeature(base, fx)

        elif "ET" in i:
            A.AddEdgeTruth(base, fx)
        elif "NT" in i:
            A.AddNodeTruth(base, fx)
        elif "GT" in i:
            A.AddGraphTruth(base, fx)
