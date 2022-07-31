

def TestImports():
    from AnalysisTopGNN.Templates import ParticleTemplate
    from AnalysisTopGNN.Templates import EventTemplate
    from AnalysisTopGNN.Templates import EventGraphTemplate
    print("PASSED TEMPLATE IMPORTS")

    from AnalysisTopGNN.Tools import VariableManager
    from AnalysisTopGNN.Tools import Notification
    from AnalysisTopGNN.Tools import Debugging
    from AnalysisTopGNN.Tools import Threading
    from AnalysisTopGNN.Tools import TemplateThreading
    from AnalysisTopGNN.Tools import RecallObjectFromString
    from AnalysisTopGNN.Tools import Metrics
    print("PASSED TOOL IMPORTS")

    from AnalysisTopGNN.Events import Event
    from AnalysisTopGNN.Events import EventDelphes
    from AnalysisTopGNN.Events import ExperimentalEvent
    from AnalysisTopGNN.Events import EventGraphs
    print("PASSED EVENT IMPORTS")

    from AnalysisTopGNN.Particles import Particles
    from AnalysisTopGNN.Particles import ExperimentalParticles
    print("PASSED PARTICLE IMPORTS")

    from AnalysisTopGNN.IO import Directories
    from AnalysisTopGNN.IO import WriteDirectory
    from AnalysisTopGNN.IO import File 
    from AnalysisTopGNN.IO import PickleObject
    from AnalysisTopGNN.IO import UnpickleObject
    from AnalysisTopGNN.IO import HDF5
    from AnalysisTopGNN.IO import ExportToDataScience
    print("PASSED IO IMPORTS")  

    from AnalysisTopGNN.Generators import GenerateDataLoader
    from AnalysisTopGNN.Generators import EventGenerator
    from AnalysisTopGNN.Generators import Optimizer
    from AnalysisTopGNN.Generators import CacheGenerators
    print("PASSED GENERATOR IMPORTS")  
    
    from AnalysisTopGNN.Plotting.Legacy import TH1F
    from AnalysisTopGNN.Plotting.Legacy import TH2F
    from AnalysisTopGNN.Plotting.Legacy import CombineHistograms
    from AnalysisTopGNN.Plotting.Legacy import CombineTGraph
    from AnalysisTopGNN.Plotting.Legacy import TGraph
    from AnalysisTopGNN.Plotting.Legacy import Graph

    from AnalysisTopGNN.Plotting import TH1F
    from AnalysisTopGNN.Plotting import CombineTH1F
    print("PASSED PLOTTING IMPORTS")  
 
    from AnalysisTopGNN.Models import BaseLineModel
    from AnalysisTopGNN.Models import BaseLineModelAdvanced
    from AnalysisTopGNN.Models import BaseLineModelEvent
    from AnalysisTopGNN.Models import BasicEdgeConvolutionBaseLine
    from AnalysisTopGNN.Models import BasicMessageBaseLine
    from AnalysisTopGNN.Models import BasicBaseLineTruthChildren
    from AnalysisTopGNN.Models import BasicBaseLineTruthJet

    from AnalysisTopGNN.Models import GraphNN
    from AnalysisTopGNN.Models import NodeConv
    from AnalysisTopGNN.Models import EdgeConv
    from AnalysisTopGNN.Models import CombinedConv
    from AnalysisTopGNN.Models import MassGraphNeuralNetwork
    print("PASSED MODEL IMPORTS")  
    
    return True 

