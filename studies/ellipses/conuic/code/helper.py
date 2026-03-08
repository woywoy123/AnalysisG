import sympy as sp

def symbol(name): return sp.symbols(name, real = True, positive = True)
def symbols(lst): return [symbol(i) for i in lst]

def prove(trg, clm):
    try: assert sp.simplify(trg - clm) == 0; print("(OK)", trg, "=", clm)
    except AssertionError: 
        print("FAILED:")
        sp.pprint(sp.simplify(trg - clm))

class particle:
    def __init__(self, name):
        self.name   = name
        self.p      = symbol("p_"    + name)
        self.mass   = symbol("m_"    + name)
        self.beta   = symbol("beta_" + name)
        self.E      = symbol("E_"    + name)

    def GetP(self, inpt):  return inpt.subs(self.p   , self.E * self.beta)
    def GetM2(self, inpt): return inpt.subs(self.mass**2, self.E**2 * (1 - self.beta**2))
    def GetBeta2(self, inpt): return inpt.subs(self.beta ** 2, 1 - self.mass**2 / self.E**2)


