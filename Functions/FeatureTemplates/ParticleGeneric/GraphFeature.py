

# ================== Kinematics ================ #
def mu(ev):
    return float(ev.mu)

def met(ev):
    return float(ev.met)

def met_phi(ev):
    return float(ev.met_phi)

def pileup(ev):
    return float(ev.pileup)

def nJets(ev):
    return float(len(ev.Jets))

def nTruthJet(ev):
    return float(len(ev.TruthJets))

# ================== Truth ================ #
def mu_actual(ev):
    return float(ev.mu_actual)

def nTops(ev):
    return float(len(ev.TruthTops))

