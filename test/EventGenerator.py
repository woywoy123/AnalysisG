from AnalysisTopGNN.Generators import EventGenerator 

def TestEventGenerator(Files):


    EvtGen = EventGenerator(Files[0])
    EvtGen.SpawnEvents()




    return True
