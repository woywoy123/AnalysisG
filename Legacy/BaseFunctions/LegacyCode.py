from BaseFunctions.VariableManager import *
from BaseFunctions.Alerting import *
from BaseFunctions.ParticlesManager import *
import multiprocessing
import numpy as np

# ______ Legacy Code! ________ #
def CreateParticles(e, pt, phi, eta, pdg = [], index = "", sig = [], flavour = []):
    Output = [] 

    for i in range(len(e)):
        P = Particle()

        P.SetKinematics(e[i], pt[i], phi[i], eta[i])
        if len(sig) == len(e):
            P.IsSignal = sig[i]

        if len(pdg) == len(e):
            P.PDGID = pdg[i]

        if len(flavour) == len(e):
            P.Flavour = flavour[i]


        if isinstance(index, str) == False:
            P.Index = index
            if index == -1:
                P.Index = i

        Output.append(P)

    return Output

# ______ Legacy Code! ________ #
class SignalSpectator(BranchVariable):
    
    def __init__(self, Tree, Branches, file_dir):
        super().__init__(file_dir, Tree, Branches)
        self.EventContainer = []
        self.EventLoop()

    def ProcessLoop(self, e, pt, phi, eta, mk, pdg):
        parent = False 
        Map = {}
        Map["All"] = []
        Map["Signal"] = []
        Map["Spectator"] = []

        params = []
        pipes = []
        for i in range(len(mk)):
            
            if isinstance(e[i], np.float32) == True:
                parent = True
                break
            
            part = CreateParticles(e[i], pt[i], phi[i], eta[i], pdg[i], i, [mk[i]]*len(eta[i]))
            for z in part:
                if z.IsSignal == 1:
                    Map["Signal"].append(z)
                if z.IsSignal == 0:
                    Map["Spectator"].append(z)
                Map["All"].append(z)

        if parent:
            part = CreateParticles(e, pt, phi, eta, pdg, -1, mk)
            for z in part:
                Map["All"].append(z)

        if parent == False:
            Map = self.ParticleParent(Map)
        return Map

    def ParticleParent(self, Map):
        
        s = {}
        for i in Map["All"]:
            try:
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
            except KeyError:
                s[i.Index] = Particle()
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
        
        for i in s:
            s[i].ReconstructFourVectorFromProducts()

        Map["FakeParents"] = s
        return Map

    def BulkRunning(self, inst, sender):
        
        EventContainer = []
        for i in inst:
            Map = self.ProcessLoop(i[0], i[1], i[2], i[3], i[4], i[5])
            EventContainer.append(Map)
        sender.send(EventContainer)


    def EventLoop(self):
        print("INFO::Entering the EventLoop Function")
            
        Params = []
        Pipe = []
        processes = []
        bundle_s = 4000
        
        for i in range(len(self.Mask)):
            inst = [self.E[i], self.Pt[i], self.Phi[i], self.Eta[i], self.Mask[i], self.PDGID[i]]
            Params.append(inst)
            
            if len(Params) == bundle_s:
                recv, send = multiprocessing.Pipe(False)
                p = multiprocessing.Process(target = self.BulkRunning, args = (Params, send, ))
                processes.append(p)
                Pipe.append(recv)
                Params = []
        
        if len(Params) != 0:
            recv, send = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target = self.BulkRunning, args = (Params, send, ))
            processes.append(p)
            Pipe.append(recv)
            Params = []
       
        for i in processes:
            i.start()

        al = Alerting(len(processes))
        for i, j in zip(processes, Pipe):
            con = j.recv()
            i.join() 
            for t in con:
                self.EventContainer.append(t)

            al.ProgressAlert() 

        print("INFO::Finished EventLoop")               
    
