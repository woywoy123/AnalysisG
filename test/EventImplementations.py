from AnalysisTopGNN.Events import EventDelphes
from AnalysisTopGNN.Events import ExperimentalEvent
from AnalysisTopGNN.Generators import EventGenerator

def TestDelphes(FileDir):
    E = EventDelphes()
    ev = EventGenerator(FileDir, Stop = 2)
    ev.EventImplementation = E
    ev.SpawnEvents()
    ev.CompileEvent(True)

    for i in ev.Events:
        event = ev.Events[i]["Delphes"]

        print(event.Jets)
        if len(event.Particle) != 0:
            return True
    
    return False

def TestExperiemental(FileDir):
    def Recursive(inp, it):
        if len(inp.Daughter) == 0:
            return
          
        it+=1
        tmp = "-"*it +"> " + str(inp.Mass_GeV)[:5] + " " + str(inp.MassDaught_GeV)[:5]
        tmp2 = "(" + str(inp.pdgid) + ") -> " + " + ".join([str(k.pdgid) for k in inp.Daughter])

        print(tmp2 + tmp)
        for k in inp.Daughter:
            if k.index > 4:
                continue
            Recursive(k, it)

    def FindLeptonTop(t):
        if abs(t.pdgid) == 24 and len([i for i in t.Daughter if abs(i.pdgid) == 24 ]) == 0:
            return [p for p in t.Daughter if abs(p.pdgid) >= 11 and abs(p.pdgid) <= 18]
        else:
            for i in t.Daughter:
                return FindLeptonTop(i)




    E = ExperimentalEvent()
    ev = EventGenerator(FileDir, Stop = 10000, Debug = False)
    ev.EventImplementation = E
    ev.SpawnEvents()
    ev.CompileEvent()
        
    n_N = 0 # Number of negative leptons
    n_P = 0 # Number of positive leptons
    n_L = 0 # Number of leptons
    n_SS = 0 # Number of events with Same Sign Dileptons
    n_LEvent = {}
    for i in ev.Events:
        event = ev.Events[i]["nominal"]
        leptons = []
        for t in event.Tops:
            #Recursive(t, 0)
            leptons += FindLeptonTop(t)
        
        tmp_L = []
        for l in leptons:
            if l.charge == 0 or abs(l.pdgid) == 15:
                continue
            tmp_L.append(l)            
            n_L += 1
            
            if l.charge < 0:
                n_N += 1
            if l.charge > 0:
                n_P += 1
        nl = len(tmp_L) 
        if nl not in n_LEvent:
            n_LEvent[nl] = 0
        n_LEvent[nl] += 1
        
        pair = [i for i in tmp_L for j in tmp_L if i != j and i.charge * j.charge > 0]
        if len(pair) != 0:
            n_SS += 1

    print("(@ Truth) Total Leptons:", n_L)
    print("(@ Truth) Total Negative:", n_P)
    print("(@ Truth) Total Positive:", n_N)
    print("(@ Truth) Total Events with Same Sign Dileptons:", n_SS)
    print("(@ Truth) n-Leptons per event:\n", n_LEvent)
        
    return True


    
