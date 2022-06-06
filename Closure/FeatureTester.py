from Functions.Event.EventGenerator import EventGenerator
from Functions.Tools.Alerting import Notification
import Functions.Event.EventGraph as EG
import Functions.FeatureTemplates.GraphFeatures as GF
import Functions.FeatureTemplates.NodeFeatures as NF
import Functions.FeatureTemplates.EdgeFeatures as EF
import inspect
import importlib
import sys

def FeatureTester(Input, EventGraph, Fx):
    nf = Notification()
    nf.Caller = "FeatureTester"
    Passed = True

    fx_m = EventGraph.__module__ 
    fx_n = EventGraph.__name__
    EG = getattr(importlib.import_module(fx_m), fx_n)
    
    Fx_m = Fx.__module__ 
    Fx_n = Fx.__name__
    Fx = getattr(importlib.import_module(Fx_m), Fx_n)
    try: 
        ev = EG(Input)
        ev.SelfLoop = True
        ev.FullyConnect = True 
        if "GraphFeatures" in Fx_m:
            ev.GraphAttr = {Fx_n : Fx}

        if "NodeFeatures" in Fx_m:
            ev.NodeAttr = {Fx_n : Fx}

        if "EdgeFeatures" in Fx_m:
            ev.EdgeAttr = {Fx_n : Fx}
        
        ev.ConvertToData()
        nf.Notify("SUCCESS (" + fx_n + "): " + Fx_m + " -> " + Fx_n)

    except AttributeError:
        fail = str(sys.exc_info()[1]).replace("'", "").split(" ")
        func = fail[0]
        attr = fail[-1]
        nf.Warning(">--------------------------------------------------------------------------------<")
        nf.Warning("FAILED (" + fx_n + "): " + Fx_m + " -> " + Fx_n + " ERROR -> " + attr)
        nf.Warning(">--------------------------------------------------------------------------------<")
        Passed = False
    return Passed

def Analyser(Files):
    
    GraphFunc = {i : j for i, j in inspect.getmembers(GF, inspect.isfunction)}
    NodeFunc = {i : j for i, j in inspect.getmembers(NF, inspect.isfunction)}
    EdgeFunc = {i : j for i, j in inspect.getmembers(EF, inspect.isfunction)}
    EventGraphs = {i : j for i, j in inspect.getmembers(EG, inspect.isclass) if i not in ["Data", "EventGraphTemplate", "Notification"]}
    for i in Files:
        ev = EventGenerator(i, Stop = 1)
        ev.SpawnEvents()
        ev.CompileEvent()
        ev = ev.Events[0]["nominal"]

        for eg, ef in EventGraphs.items():
            for g_n, g_f in GraphFunc.items():
                FeatureTester(ev, ef, g_f)
            for n_n, n_f in NodeFunc.items():
                FeatureTester(ev, ef, n_f)
            for e_n, e_f in EdgeFunc.items():
                FeatureTester(ev, ef, e_f)
        

    return True

