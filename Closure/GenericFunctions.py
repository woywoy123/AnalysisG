def Recursive(inp1, inp2):
    if isinstance(inp1, dict):
        if len(inp1) != len(inp2):
            print(inp1, inp2)
            exit()
        for key in inp1:
            Recursive(inp1[key], inp2[key])
    elif isinstance(inp1, list):
        for key1, key2 in zip(inp1, inp2):
            Recursive(key1, key2)
    else:
        if inp1 != inp2:
            try:
                if "_store" in inp1.__dict__:
                    import torch
                    for i in inp1.__dict__["_store"].keys():
                        a = inp1.__dict__["_store"][i]
                        b = inp2.__dict__["_store"][i]
                        if torch.all( a == b ):
                            continue
                        else:
                            print(a, b)
                    return 
            except:
                pass
            if "function" in str(type(inp1)):
                return 

            if isinstance(inp1, float) and isinstance(inp2, float):
                r = round(inp1/inp2)
                if r == 1:
                    return 
            print(inp1, inp2, inp1 == inp2)
            for i, j in zip(inp1.__dict__, inp2.__dict__):
                p1, p2 = inp1.__dict__[i], inp2.__dict__[j]
                if p1 == p2:
                    continue
                print(" > ", p1, p2, p1 == p2, type(p1), type(p2))
            print(inp1.iter,inp2.iter) 
            return False


def CompareObjects(in1, in2):
    key_t = set(list(in1.__dict__.keys()))
    key_r = set(list(in2.__dict__.keys()))
    d = key_t ^ key_r
    
    if len(list(d)) != 0:
        print("!!!!!!!!!!!!!!!!Variable difference: ", d)
    
    for i, j in zip(list(key_t), list(key_r)):
        if Recursive(in1.__dict__[i], in2.__dict__[j]) == None:
            continue
        else:
            print("----> ", i, j)
            exit()

def CacheEventGenerator(Stop, Dir, Name, Cache):
    from Functions.IO.IO import UnpickleObject, PickleObject   
    from Functions.Event.EventGenerator import EventGenerator

    if Cache:
        ev = EventGenerator(Dir, Stop = Stop)
        ev.SpawnEvents()
        ev.CompileEvent(SingleThread = False)
        PickleObject(ev, Name) 
    return UnpickleObject(Name)


def CreateEventGeneratorComplete(Stop, Files, Name, CreateCache, NameOfCaller):
    from Functions.IO.IO import UnpickleObject, PickleObject   
    from Functions.Event.EventGenerator import EventGenerator 

    out = []
    for i, j in zip(Files, Name):
        if CreateCache:
            ev = EventGenerator(i, Stop = Stop)
            ev.SpawnEvents()
            ev.CompileEvent(SingleThread = False)
            PickleObject(ev, j, Dir = "_Pickle/" +  NameOfCaller) 
        out.append(UnpickleObject(j, Dir = "_Pickle/" + NameOfCaller))
    return out

def CreateDataLoaderComplete(Files, Level, Name, CreateCache, NameOfCaller = None):
    from Functions.IO.IO import UnpickleObject, PickleObject   
    if CreateCache:

        import Functions.FeatureTemplates.EdgeFeatures as ef
        import Functions.FeatureTemplates.NodeFeatures as nf
        import Functions.FeatureTemplates.GraphFeatures as gf
        from Functions.Event.DataLoader import GenerateDataLoader

        DL = GenerateDataLoader()

        # Edge Features 
        DL.AddEdgeFeature("dr", ef.d_r)
        DL.AddEdgeFeature("mass", ef.mass)       
        DL.AddEdgeFeature("signal", ef.Signal)
 
        # Node Features 
        DL.AddNodeFeature("eta", nf.eta)
        DL.AddNodeFeature("pt", nf.pt)       
        DL.AddNodeFeature("phi", nf.phi)      
        DL.AddNodeFeature("energy", nf.energy)
        DL.AddNodeFeature("signal", nf.Signal)
        
        # Graph Features 
        DL.AddGraphFeature("mu", gf.Mu)
        DL.AddGraphFeature("m_phi", gf.MissingPhi)       
        DL.AddGraphFeature("m_et", gf.MissingET)      
        DL.AddGraphFeature("signal", gf.Signal)       


        # Truth Stuff 
        DL.AddEdgeTruth("Topology", ef.Signal)
        DL.AddNodeTruth("NodeSignal", nf.Signal)
        DL.AddGraphTruth("GraphMuActual", gf.MuActual)
        DL.AddGraphTruth("GraphEt", gf.MissingET)
        DL.AddGraphTruth("GraphPhi", gf.MissingPhi)
        DL.AddGraphTruth("GraphSignal", gf.Signal)

        DL.SetDevice("cuda")
        for i in Files:
            if NameOfCaller != None:
                ev = UnpickleObject(NameOfCaller + "/" + i)
            else:
                ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, True, True)
        DL.MakeTrainingSample(0)
        PickleObject(DL, Name)
    return UnpickleObject(Name)






