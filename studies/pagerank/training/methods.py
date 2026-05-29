from AnalysisG.graphs.bsm_4tops.graph_bsm_4tops import *
from AnalysisG.events.bsm_4tops.event_bsm_4tops import *
from training.config import  *
from AnalysisG.models import *

def RaiseError(msg, term = True):
    if term: msg = "ERROR [TERMINATED]: " + msg
    else: msg = "WARNING: " + msg
    print(msg)
    if term: exit()

def event_method(instance, get_ev = False):
    if instance.name == ""        : return SwitchType(get_ev, None)
    if instance.name == "bsm4tops": return SwitchType(get_ev, BSM4Tops)
    RaiseError("INVALID EVENT", True)

def graph_method(instance, get_gr = False):
    if instance.name == "graphtruthjets"    : return SwitchType(get_gr, GraphTruthJets    )
    if instance.name == "graphtruthjetsnonu": return SwitchType(get_gr, GraphTruthJetsNoNu)
    if instance.name == "graphjets"         : return SwitchType(get_gr, GraphJets         )
    if instance.name == "graphjetsnonu"     : return SwitchType(get_gr, GraphJetsNoNu     )
    if instance.name == "graphdetectorlep"  : return SwitchType(get_gr, GraphDetectorLep  )
    if instance.name == "graphdetector"     : return SwitchType(get_gr, GraphDetector     )
    RaiseError("INVALID GRAPH", True)

def model_method(name, get_md = False):
    if name == "recursivegraphneuralnetwork": return SwitchType(get_md, RecursiveGraphNeuralNetwork())
    if name == "grift"                      : return SwitchType(get_md, Grift())
    RaiseError("INVALID MODEL", True)

def optimizer_method(name):
    if   name == "adam"   : return name
    elif name == "adagrad": return name
    elif name == "adamw"  : return name
    elif name == "rmsprop": return name
    elif name == "sgd"    : return name
    elif name == "lbfgs"  : RaiseError("NOT IMPLEMENTED", True)
    RaiseError("INVALID OPTIMIZER", True)

def scheduler_method(instance):
    if instance.name == "steplr"                    : return True
    if instance.name == "reducelronplateauscheduler": return True
    if instance.name == "lrscheduler"               : return True
    RaiseError("INVALID SCHEDULER", True)

def loss_method(name):
    if name == "bce"                      : return name
    if name == "bcewithlogits"            : return name
    if name == "cosineembedding"          : return name
    if name == "crossentropyloss"         : return name
    if name == "ctc"                      : return name
    if name == "hingeembedding"           : return name
    if name == "huber"                    : return name
    if name == "kldiv"                    : return name
    if name == "l1"                       : return name
    if name == "marginranking"            : return name
    if name == "mse"                      : return name
    if name == "multilabelmargin"         : return name
    if name == "multilabelsoftmargin"     : return name
    if name == "multimargin"              : return name
    if name == "nll"                      : return name
    if name == "poissonnll"               : return name
    if name == "smoothl1"                 : return name
    if name == "softmargin"               : return name
    if name == "tripletmargin"            : return name
    if name == "tripletmarginwithdistance": return name
    RaiseError("INVALID LOSS", True)

def loss_string_method(name, val):
    if name == ""         : return name 
    if name == "none"     : return name
    if name == "sum"      : 
        if isinstance(val, bool): return name
        RaiseError("INVALID LOSS TYPE [sum]: bool")

    if name == "smoothing": 
        if isinstance(val, float): return name
        RaiseError("INVALID LOSS TYPE [smoothing]: float")

    if name == "mean"     : return name
    if name == "swap"     : return name
    if name == "full"     : return name
    if name == "batchmean": return name
    if name == "target"   : return name
    if name == "zeroinf"  : return name
    if name == "ignore"   : return name
    if name == "blank"    : return name
    if name == "margin"   : return name
    if name == "beta"     : return name
    if name == "eps"      : return name
    if name == "weight"   : return name
    RaiseError("INVALID LOSS STRING", True)


