




# =========== TRUTH ========= #
def PDGID(a):
    return a.pdgid

def TopsMerged(a):
    ta = [i.Index for i in a.Decay if i.Index != -1] 
    if a.Type == "el" or a.Type == "mu":
        ta.append(a.Index)
    return len(list(set(ta)))

def FromRes(a):
    return a.FromRes
