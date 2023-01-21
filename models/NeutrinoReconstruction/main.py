from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject

#direc = "/home/tnom6927/Downloads/samples/tttt/QU_0.root"
#Ana = Analysis()
#Ana.InputSample("bsm1000", direc)
#Ana.Event = Event
#Ana.EventCache = True
#Ana.DumpPickle = True 
#Ana.Launch()
#
#
#for i in Ana:
#    ev = i.Trees["nominal"]
#    
#    it = 0
#    for t in ev.Tops:
#        it += 1 if t.DecayLeptonically() else 0
#    if it == 1:
#        PickleObject(ev, "TMP")
#        break

ev = UnpickleObject("TMP")
singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]
singlelepton = {abs(i.pdgid) : i for i in singlelepton}
b = singlelepton[5]
muon = singlelepton[13]
nu = singlelepton[14]

mW = 80.385*1000 # MeV : W Boson Mass
mT = 172.5*1000  # MeV : t Quark Mass
mN = 0           # GeV : Neutrino Mass

from BaseFunctionsTests import *

#print("---- Comparing the Four Vectors ----")
#print("-> b-quark")
#TestFourVector(b)
#print("-> muon")
#TestFourVector(muon)
#print("-> neutrino")
#TestFourVector(nu)
#
#print("---- Comparing CosTheta and SinTheta ----")
#TestCosTheta(b, muon)
#
#print("---- Comparing x0 ----")
#print("b + W:")
#Testx0(mT, mW, b)
#
#print("nu + mu:")
#Testx0(mW, mN, muon)
#
#print("----- Comparing Beta -----")
#print("b")
#TestBeta(b)
#
#print("muon")
#TestBeta(muon)
#
#print("------ Comparing SxSy --------")
#TestSValues(b, muon, mT, mW, mN)
#
#print("------- Comparing Eps W(_) Omega -------")
#TestIntermediateValues(b, muon, mT, mW, mN)
#TestEps_W_Omega(b, muon, mW, mN)
#
#print("------- Comparing x, y, Z2 --------")
#TestxyZ2(b, muon, mT, mW, mN)

#print("-------- Comparing S2 V0 --------")
#TestS2V0(1, 2, 3, 4, ev.met_phi, ev.met)

print("--------- Comparing deltaNu --------")
TestR_T(ev.met_phi, ev.met, b, muon, mT, mW, mN)


#    @property
#    def R_T(self ):
#        '''Rotation from F coord.to laboratory coord.'''
#        b_xyz = self.b.X(), self.b.Y(), self.b.Z()
#        R_z = R(2, -self.mu.Phi ())
#        R_y = R(1, 0.5* math.pi - self.mu.Theta ())
#        R_x = next(R(0,-math.atan2(z,y)) for x,y,z in (R_y.dot(R_z.dot(b_xyz )) ,))
#        return R_z.T.dot(R_y.T.dot(R_x.T))
#    @property
#    def H_tilde(self ):
#        '''Transformation of t=[c,s ,1] to p_nu: F coord.'''
#        x1 , y1 , p = self.x1 , self.y1 , self.mu.P()
#        Z, w, Om = self.Z, self.w, math.sqrt(self.Om2)
#        return np.array ([[ Z/Om , 0, x1 -p], [w*Z/Om , 0, y1], [ 0, Z, 0]])
#    @property
#    def H(self ):
#        '''Transformation of t=[c,s ,1] to p_nu: lab coord.'''
#        return self.R_T.dot(self.H_tilde)



#def R(axis , angle ):
#    '''Rotation matrix about x(0),y(1), or z(2) axis '''
#    c, s = math.cos(angle), math.sin(angle)
#    R = c * np.eye (3)
#    for i in [-1, 0, 1]:
#        R[(axis -i)%3, (axis+i)%3] = i*s + (1 - i*i)
#    return R
