import FeatureTemplates.Generic.EdgeFeature as ef
import FeatureTemplates.Generic.NodeFeature as nf
import FeatureTemplates.Generic.GraphFeature as gf

import FeatureTemplates.TruthTopChildren.NodeFeature as tc_nf

import FeatureTemplates.TruthJet.EdgeFeature as tj_ef
import FeatureTemplates.TruthJet.NodeFeature as tj_nf
import FeatureTemplates.TruthJet.GraphFeature as tj_gf
from AnalysisTopGNN.Tools.ModelTesting import AddFeature

def TruthJetsFeatures(Analysis):
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

    for i in Features:
        base = "_".join(i.split("_")[1:])
        fx = Features[i]
        
        if "EF" in i:
            Analysis.AddEdgeFeature(base, fx)
        elif "NF" in i:
            Analysis.AddNodeFeature(base, fx)
        elif "GF" in i:
            Analysis.AddGraphFeature(base, fx)

        elif "ET" in i:
            Analysis.AddEdgeTruth(base, fx)
        elif "NT" in i:
            Analysis.AddNodeTruth(base, fx)
        elif "GT" in i:
            Analysis.AddGraphTruth(base, fx)
    return Features
 
