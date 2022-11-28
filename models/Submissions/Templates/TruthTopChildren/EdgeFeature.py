def edge(a, b):
    if a.index == b.index:
        return 1
    return 0

def ResEdge(a, b):
    p1 = a.Parent
    p2 = b.Parent
    if sum([1 for i in p1 for j in p2 if i.FromRes == j.FromRes]) > 0:
        return 1
    return 0
    

