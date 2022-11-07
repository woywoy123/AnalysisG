from AnalysisTopGNN.Generators import Analysis
from ExperimentalEvent import EventExperimental
from AnalysisTopGNN.IO import UnpickleObject

Direc = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/ttbarFull/"


#root = UnpickleObject("UNTITLED/Tracers/ttbar.pkl")
#root._locked = False
#root.ClearEvents()
#print(root._locked)
#expl = list(root.ROOTFiles.values())[0]
#expl._lock = False
#expl.ClearEvents() 
#print(expl.EventMap)
#print(list(root._EventMap.values())[0])


Ana = Analysis()
Ana.InputSample("ttbar", Direc)
Ana.EventCache = True
Ana.DumpPickle = True
Ana.EventStop = 10
Ana.chnk = 2
Ana.Threads = 5
Ana.Event = EventExperimental
Ana.Launch()



def Recur(inpt):
    final = []
    for k in inpt.Children:
        if abs(k.pdgid) == 24:
            return [x for x in inpt.Children]
        Recur(k)
    return final


for i in Ana:
    event = i.Trees["nominal"]

    collect = {}
    tops = event.Tops
    
    for t in tops:
        al = Recur(t.Children[0])
        print([k.pdgid for k in al])
        print(sum(al).CalculateMass())
