



# ============= Truth =================== #
def Index(a, b):
    ta = [i.Index for i in a.Decay]
    tb = [i.Index for i in b.Decay]
    
    if a.Type == "el" or a.Type == "mu":
        ta.append(a.Index)
    if b.Type == "el" or b.Type == "mu":
        tb.append(b.Index)

    if len(ta) == 0:
        return 0
    if len(tb) == 0:
        return 0
    
    x = [p for p in ta for j in tb if p == j]
    if len(x) > 0:
        return  1
    else:
        return 0


