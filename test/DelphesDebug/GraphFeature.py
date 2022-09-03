def nTops(ev):
    return float(len(ev.Tops))

def SignalSample(ev):
    if len(ev.Tops) == 2:
        return 1
    else:
        return 0
