
# =========== TRUTH ========= #
def PDGID(a):
    return a.pdgid

def TopsMerged(a):
    ta = [i.Index for i in a.Decay] 
    if a.Type == "el" or a.Type == "mu":
        ta.append(a.Index)
    ta = [i for i in ta if i != -1]
    if len(list(set(ta))) > 1:
        return 1
    return 0

def FromTop(a):
    ta = [i.Index for i in a.Decay] 
    if a.Type == "el" or a.Type == "mu":
        ta.append(a.Index)
    ta = [i for i in ta if i != -1]
    if len(list(set(ta))) > 0:
        return 1
    return 0
