def signal(ev):
    try: return int(len([i for i in ev.Tops if i.FromRes == 1]) == 2)
    except: 0

def ntops(ev):
    try: t = len(ev.Tops)
    except: return 0
    return 4 if t > 4 else t

def n_nu(ev):
    try: return sum([c.is_nu for c in ev.TopChildren])
    except: return 0

def n_jets(ev):
    try: return len(ev.TruthJets)
    except: return 0

def n_lep(ev):
    try: return len([k for k in ev.TopChildren if k.is_lep])
    except: return 0

def met(ev): return ev.met
def phi(ev): return ev.met_phi
