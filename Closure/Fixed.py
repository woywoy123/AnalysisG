#####
#               Truth matching for the GNN sample
#####

class GenericParticle:

    def __init__(self):
        
        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"
        
        self.Daughter = []
        self.Parent = []
    
    def SelfMass(self):
        self.CalculateMass()
    
    def MassFromChild(self):
        self.CalculateMass(self.Daughter, "MassDaught")

class Top(GenericParticle):

    def __init__(self):
        self.Type = "top"
        
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.index = self.Type + "_index"

        GenericParticle.__init__(self)


class TopChild(GenericParticle):

    def __init__(self):
        self.Type = "children"
        
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        self.index = self.Type + "_index"
        self.topindex = self.Type + "_TopIndex"
        self.jetindex = self.Type + "_JetIndex"
        self.truthjetindex = self.Type + "_TruthJetIndex"

        GenericParticle.__init__(self)

def PickleObject(obj, filename):
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"
   
    outfile = open(filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename):
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"

    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj

class Event:
    def __init__(self):
        self.Tops = {}
        self.TopChildren = {}
        self.N_children = None
        self.N_tops = None
    
    def ProcessInput(self, inp, leaf):
        def Switch(key):
            if key == "children_":
                return TopChild()
            elif key == "top_":
                return Top()

        def SearchKeys(obj, rej):
            for key in obj.__dict__:
                val = obj.__dict__[key]
                if isinstance(val, str) == False:
                    continue
                if val == rej:
                    return key

        def Fill(dic, key, data):
            if key not in data[1]:
                return
            for i in range(len(data[0])):
                if i not in dic:
                    dic[i] = Switch(key)
                val = data[0][i]
                x = SearchKeys(dic[i], data[1])
                if x == None:
                    continue
                setattr(dic[i], x, val)
            return dic

        for i in zip(inp, leaf):
            v = Fill(self.TopChildren, "children_", i)
            if v != None:
                self.TopChildren |= v
            v = Fill(self.Tops, "top_", i)
            if v != None:
                self.Tops |= v
        self.TruthMatching()

    def TruthMatching(self):
        collect = {}
        for i in self.TopChildren:
            t = self.TopChildren[i]
            if t.topindex not in collect:
                collect[t.topindex] = {}
            if t.index not in collect[t.topindex]:
                collect[t.topindex][t.index] = []
            collect[t.topindex][t.index].append(t)
        
            if t.index > 0:
                collect[t.topindex][t.index-1][-1].Daughter.append(t)
        
        self.Tops = [] 
        for i in collect:
            self.Tops.append(collect[i][0][0])
        self.TopChildren = [self.TopChildren[i] for i in self.TopChildren]


def FindLeptonTop(t):
    if abs(t.pdgid) == 24 and len([i for i in t.Daughter if abs(i.pdgid) == 24 ]) == 0:
        return [p for p in t.Daughter if abs(p.pdgid) >= 11 and abs(p.pdgid) <= 18]
    else:
        for i in t.Daughter:
            return FindLeptonTop(i)







import uproot 
import pickle

Input = "/CERN/CustomAnalysisTopOutputSameSignDilepton/Merger/QU_0.root"
Tree = "nominal"
data = uproot.open(Input)[Tree]
nEvents = 100



## Designate arrays for the lepton children information
#p_id = data["children_pdgid"].array()
#p_sign = data["children_charge"].array()
#p_index = data["children_index"].array()
#p_topIndex = data["children_TopIndex"].array()
#p_eta = data["children_eta"].array()
#p_phi = data["children_phi"].array()
#
##### ====== This does some caching! So you dont have to always rerun Uproot..... (much quicker)
#Stuff = [p_id, p_sign, p_index, p_topIndex, p_eta, p_phi]
#PickleObject(Stuff, "DEBUG")
Stuff = UnpickleObject("DEBUG")

p_id = Stuff[0]
p_sign = Stuff[1]
p_index = Stuff[2]
p_topIndex = Stuff[3]
p_eta = Stuff[4]
p_phi = Stuff[5]

#### ======
    
# Find leptons meeting criteria of being the second decay product and fill array
leptons_list = []
for event in range(len(p_id)):
    # Start a new list for every event with the first entry being the index of that event
    current_event = [event]
    # For every particle in the current event
    

    t, t, t, t -> (b W+), (b W-), (b W+), (b W-)

    t_0 -> b e+
    t_1 -> b q qbar
    t_2 -> b mu+
    t_3 -> b e-

    print("____NEW EVENT____")
    for particle in range(len(p_id[event])):
        # Keep index of particles in the event if they are a lepton that is the second decay product
        if (abs(p_id[event, particle]) == 11 or abs(p_id[event, particle]) == 13) and p_index[event, particle] == 2:
            current_event.append(particle)
            
        if abs(p_id[event, particle]) == 13 and p_index[event, particle] < 4:
            print(abs(p_id[event, particle]))


    print(current_event)
    # Append this event to the list of events if it meets the same-sign dilepton criteria
    if len(current_event) == 3 and p_sign[event, current_event[1]] == p_sign[event, current_event[2]]:
        leptons_list.append(current_event)

# leptons_list now includes one array per same-sign, dilepton event with entries corresponding to [event #, index of lepton 1, index of lepton 2]

































## INSTRUCTIONS (NOTES):
## -> This creates compiled events which can be easily navigated around. 
## -> Lets say you want to navigate the particle's decay chain, you can just grab a particle from the 'Event' object by Event.<Something>
##    Now you basically grab the first particle e.g. Top and do the following; Event.Tops[0].Daughter
##    This will return the particles the top decay's into. You can go even further down the decay chain by grabbing the daughters, 
##    And iterating over Daughter = Event.Tops[0].Daughter, where each Daughter[i] has the key attribute <Daughter>
## -> For jets the above algorithm can basically be recycled, all you need to do, is use the 'GenericParticle' template and copy the 'Top' class
##    you just need to make sure that self.Type is the prefix (name before '_') and the remaining is some variable like PT or charge etc.
#
#MinimalLeaves = [l for i in [Top(), TopChild()] for l in i.__dict__.values() if isinstance(l, str) and "_" in l]
#Stuff = {}
#for l in MinimalLeaves:
#    Stuff[l] = root[l].array()
#    print(l, "....... done")
#
#
##### ====== This does some caching! So you dont have to always rerun Uproot..... (much quicker)
#PickleObject(Stuff, "DEBUG")
#Stuff = UnpickleObject("DEBUG")
##### ======
#
#EventDict = {}
#it = 0
#for ev in zip(*Stuff.values()):
#    event = Event()
#    event.ProcessInput(ev, Stuff.keys()) 
#    EventDict[it] = event
#
#    if it > nEvents and nEvents != -1:
#        break
#    it+=1
#    print("Compiled Event:", it)
#
#n_N = 0 # Number of negative leptons
#n_P = 0 # Number of positive leptons
#n_L = 0 # Number of leptons
#n_SS = 0 # Number of events with Same Sign Dileptons
#n_LEvent = {}
#for i in EventDict:
#    event = EventDict[i]
#    
#    leptons = []
#    for t in event.Tops:
#        leptons += FindLeptonTop(t)
#    
#    tmp_L = []
#    for l in leptons:
#        if l.charge == 0 or abs(l.pdgid) == 15:
#            continue
#        tmp_L.append(l)            
#        n_L += 1
#        
#        if l.charge < 0:
#            n_N += 1
#        if l.charge > 0:
#            n_P += 1
#    nl = len(tmp_L) 
#    if nl not in n_LEvent:
#        n_LEvent[nl] = 0
#    n_LEvent[nl] += 1
#    
#    pair = [i for i in tmp_L for j in tmp_L if i != j and i.charge * j.charge > 0]
#    if len(pair) != 0:
#        n_SS += 1
#
#print("(@ Truth) Total Leptons:", n_L)
#print("(@ Truth) Total Negative:", n_P)
#print("(@ Truth) Total Positive:", n_N)
#print("(@ Truth) Total Events with Same Sign Dileptons:", n_SS)
#print("(@ Truth) n-Leptons per event:\n", n_LEvent)
        
