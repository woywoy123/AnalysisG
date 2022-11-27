def SignalEvent(ev):
    if sum([i.FromRes for i in ev.Tops]) == 2:
        return 1
    return 0

def NTops(ev):
    return float(len(ev.Tops))
