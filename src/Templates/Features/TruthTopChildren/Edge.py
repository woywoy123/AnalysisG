def edge(a, b):
    if a.index == b.index: return 1
    return 0

def ResEdge(a, b):
    p = a.Parent + b.Parent
    p = list(set(p))
    if len(p) == 1: return p[0].FromRes
    if len(p) != 2: return 0
    return p[0].FromRes * p[1].FromRes

def LepEdge(a, b):
    p = a.Parent + b.Parent
    p = list(set(p)) 
    if len(p) > 1: return 0
    l = len([i for i in p[0].Children if i.is_lep or i.is_nu]) 
    return 0 if l == 0 else 1

