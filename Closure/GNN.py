from Functions.Event.Event import EventGenerator
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.GNN.GNN import GNN_Model


# Closure test for the GNN being implemented in this codebase

def TestSimple4TopGNN():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"

    #ev = EventGenerator(dir)
    #ev.SpawnEvents() 
    #ev.CompileEvent(particle = "TruthTops")
    #ev = ev.Events
    #PickleObject(ev, "ONLYTOPS")
    ev = UnpickleObject("ONLYTOPS")
    
    G = GNN_Model()
    for i in ev:
        e = ev[i]["nominal"] 
        for t in e.TruthTops:
            print(t)

