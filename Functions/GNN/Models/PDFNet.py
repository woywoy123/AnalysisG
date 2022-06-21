import torch 
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh, Sigmoid
import torch.nn.functional as F

class BasePDFNetEncode(torch.nn.Module):
    
    def __init__(self, nodes = 256):
        super().__init__()
        self._mlp = Seq(Linear(1, nodes), 
                        Linear(nodes, nodes*4))
    def forward(self, kin):
        return self._mlp(kin)


class BasePDFNetDecode(torch.nn.Module):
    
    def __init__(self, nodes = 256):
        super().__init__()
        self._mlp = Seq(Linear(nodes*4, nodes), 
                        Linear(nodes, 1))
    
    def forward(self, kin):
        return self._mlp(kin)

import LorentzVector as LV
class PDFNetTruthChildren(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._nodes = 256
        
        self._PxEnc = BasePDFNetEncode()
        self._PxDec = BasePDFNetDecode()

        self._PyEnc = BasePDFNetEncode()
        self._PyDec = BasePDFNetDecode()

        self._PzEnc = BasePDFNetEncode()
        self._PzDec = BasePDFNetDecode()

        self._EnEnc = BasePDFNetEncode()
        self._EnDec = BasePDFNetDecode()

        
        self.L_eta = "MSEL"
        self.O_eta = None
        self.C_eta = False

        self.L_phi = "MSEL"
        self.O_phi = None
        self.C_phi = False

        self.L_pT = "MSEL"
        self.O_pT = None
        self.C_pT = False

        self.L_energy = "MSEL"
        self.O_energy = None
        self.C_energy = False

    def forward(self, N_eta, N_energy, N_pT, N_phi):
        
        # Convert To Cartesian and encode 
        P_cart_tru = LV.TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
        Px_enc = self._PxEnc(P_cart_tru[:, 0].view(-1, 1))
        Py_enc = self._PyEnc(P_cart_tru[:, 1].view(-1, 1))
        Pz_enc = self._PzEnc(P_cart_tru[:, 2].view(-1, 1))
        E_enc =  self._EnEnc(P_cart_tru[:, 3].view(-1, 1))
        
        # Decode the MLP and convert back to rapidity
        P_cart_rec = torch.cat([self._PxDec(Px_enc), self._PyDec(Py_enc), self._PzDec(Pz_enc), self._EnDec(E_enc)], dim = 1)
        P_mu = LV.TensorToPtEtaPhiE(P_cart_tru) #P_cart_rec)

        print(P_mu, torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))


        self.O_pT =     P_mu[:, 0].view(-1, 1)
        self.O_eta =    P_mu[:, 1].view(-1, 1)
        self.O_phi =    P_mu[:, 2].view(-1, 1)
        self.O_energy = P_mu[:, 3].view(-1, 1)
        
        delta_cart = P_cart_rec - P_cart_tru

        return self.O_eta, self.O_energy, self.O_phi, self.O_pT




















