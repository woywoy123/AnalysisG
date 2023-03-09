def edgeTop(a, b):
    mut = [1 for i in a.index if i in b.index and i > -1]
    return 1 if sum(mut) > 0 else 0

def edgeChild(a, b):
    mut = [1 for i in a.Parent for j in b.Parent if i == j]
    return 1 if sum(mut) > 0 else 0   

def edgeRes(a, b):
    c1 = sum([k.FromRes for i in a.Parent for k in i.Parent])
    c2 = sum([k.FromRes for i in b.Parent for k in i.Parent])
    
    if c1 != 0 and c2 != 0:
        return 1
    return 0
