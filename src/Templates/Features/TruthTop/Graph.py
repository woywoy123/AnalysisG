def signal(ev): 
    try: return len([i for i in ev.Tops if i.FromRes == 1]) == 2
    except: 0
def ntops(ev): return len(ev.Tops)

def met(ev): return ev.met
def phi(ev): return ev.met_phi
