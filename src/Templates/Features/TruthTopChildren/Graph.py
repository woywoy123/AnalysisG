def signal(ev): 
    try: return len([i for i in ev.Tops if i.FromRes == 1]) == 2
    except: False

def ntops(ev): 
    try: return len(ev.Tops)
    except: return 0

def n_nu(ev): 
    try: return sum([c.is_nu for c in ev.TopChildren])
    except: return 0

def met(ev): return ev.met
def phi(ev): return ev.met_phi
