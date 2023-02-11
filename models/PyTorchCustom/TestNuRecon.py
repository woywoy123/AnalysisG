from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
import PyC.NuSol.Tensors as NuT
import PyC.NuSol.CUDA as NuC
from Checks import *
from NeutrinoSolutionDeconstruct import *
from time import time
torch.set_printoptions(precision=20)
torch.set_printoptions(linewidth = 120)

def ParticleCollectors(ev):
    t1 = [ t for t in ev.Tops if t.LeptonicDecay][0]
    t2 = [ t for t in ev.Tops if t.LeptonicDecay][1]
    
    out = []
    prt = { abs(p.pdgid) : p for p in t1.Children }
    b = prt[5]
    lep = [prt[i] for i in [11, 13, 15] if i in prt][0]
    nu = [prt[i] for i in [12, 14, 16] if i in prt][0]
    out.append([b, lep, nu, t1])
    
    prt = { abs(p.pdgid) : p for p in t2.Children }
    b = prt[5]
    lep = [prt[i] for i in [11, 13, 15] if i in prt][0]
    nu = [prt[i] for i in [12, 14, 16] if i in prt][0]
    out.append([b, lep, nu, t2])

    return out

#Ana = Analysis()
#Ana.InputSample("bsm4t-1000", "/home/tnom6927/Downloads/samples/Dilepton/ttH_tttt_m1000")
#Ana.Event = Event
#Ana.EventStop = 100
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.chnk = 100
#Ana.VerboseLevel = 2
#Ana.Launch()
#
#it = 0
#vl = {"b" : [], "lep" : [], "nu" : [], "ev" : [], "t" : []}
#for i in Ana:
#    ev = i.Trees["nominal"]
#    tops = [ t for t in ev.Tops if t.LeptonicDecay]
#
#    if len(tops) == 2:
#        k = ParticleCollectors(ev)
#        vl["b"].append(  [k[0][0], k[1][0]])
#        vl["lep"].append([k[0][1], k[1][1]])
#        vl["nu"].append( [k[0][2], k[1][2]])
#        vl["t"].append(  [k[0][3], k[1][3]])
#        vl["ev"].append(ev)
#        it+=1
#    
#    if it == 100:
#        break
#    
#PickleObject(vl, "TMP")
vl = UnpickleObject("TMP")
T = SampleTensor(vl["b"], vl["lep"], vl["ev"], vl["t"], "cuda", [[100, 0], [0, 100]])
R = SampleVector(vl["b"], vl["lep"], vl["ev"], vl["t"])
print(len(vl["b"]))

its = 10000
#diff = [[], []]
#for t in range(its):
#    t1 = time()
#    t_sol = NuT.Nu(T.b, T.mu, T.met, T.phi, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN)
#    t2 = time()
#    diff1 = t2 - t1 
#    diff[0].append(diff1)
#
#for t in range(its):   
#    t1 = time()
#    t_solC = NuC.Nu(T.b, T.mu, T.met, T.phi, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN)
#    t2 = time()
#    diff2 = t2 - t1
#    diff[1].append(diff2)
#
#print(sum(diff[0]), sum(diff[1]))
#print(AssertEquivalenceRecursive(t_sol.tolist(), t_solC.tolist()))
#print("--- Testing Performance Between C++ and CUDA of Nu ---")
#print("Speed Factor (> 1 is better): ", (sum(diff[0])) / sum(diff[1]))
#
#diff = [[], []]
#for t in range(its):
#    t1 = time()
#    t_sol = NuT.NuNu(T.b, T.b_, T.mu, T.mu_, T.met, T.phi, T.mT, T.mW, T.mN)
#    t2 = time()
#    diff1 = t2 - t1 
#    diff[0].append(diff1)
#
#for t in range(its):   
#    t1 = time()
#    t_solC = NuC.NuNu(T.b, T.b_, T.mu, T.mu_, T.met, T.phi, T.mT, T.mW, T.mN)
#    t2 = time()
#    diff2 = t2 - t1
#    diff[1].append(diff2)
#
#print(sum(diff[0]), sum(diff[1]))
#print(AssertEquivalenceRecursive(t_sol.tolist(), t_solC.tolist()))
#print("--- Testing Performance Between C++ and CUDA of NuNu ---")
#print("Speed Factor (> 1 is better): ", (sum(diff[0])) / sum(diff[1]))

t_sol = NuT.Nu(T.b, T.mu, T.met, T.phi, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN)
_sol = NuC.Nu(T.b, T.mu, T.met, T.phi, T.Sxx, T.Sxy, T.Syx, T.Syy, T.mT, T.mW, T.mN)
#print(t_sol)

t_sol = NuT.NuNu(T.b, T.b_, T.mu, T.mu_, T.met, T.phi, T.mT, T.mW, T.mN)
#if AssertEquivalenceRecursive(t_sol, _sol) == False:
#    print("Not EQUAL!!!!")
for r, t in zip(R, T):
    b, mu = r[0], r[1]
    _b, _mu = r[2], r[3]
    met_x, met_y = r[4], r[5]
    mT, mW, mNu = r[6], r[7], r[8]
   
    tb_, tmu_ = t[0], t[1]
    t_b_, t_mu_ = t[2], t[3]
    t_met, t_phi = t[4], t[5]
    t_mT, t_mW, t_mNu = t[6], t[7], t[8]
    sxx, sxy = T.Sxx[T.it], T.Sxy[T.it]
    syx, syy = T.Syx[T.it], T.Syy[T.it]

    sol = singleNeutrinoSolution(b, mu, (met_x, met_y), [[100, 0], [0, 100]], mW**2, mT**2)
    t_sol = NuT.Nu(tb_, tmu_, t_met, t_phi, sxx, sxy, syx, syy, t_mT, t_mW, t_mNu)
    if AssertEquivalenceRecursive(sol.V0, t_sol.tolist()[0]) == False:
        print("------ Single -----")
        print("Not EQUAL!!!!")  
        print(t_sol)
        print("")
        print(sol.V0)
    
    try:
        sol = doubleNeutrinoSolutions((b, _b), (mu, _mu), (met_x, met_y), mW**2, mT**2)
    except:
        continue
    t_sol = NuT.NuNu(tb_, t_b_, tmu_, t_mu_, t_met, t_phi, t_mT, t_mW, t_mNu)
    
    print("")
 
    if AssertEquivalenceRecursive(sol.V0, t_sol.tolist()[0]) == False:
        print("")
        print("------ Double -----")
        print("Not EQUAL!!!!")
        print(t_sol[0])
        print("")
        print(sol.V0)
        exit()

    
   
