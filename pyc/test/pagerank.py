from common import *
from pyc import *
import torch
torch.set_printoptions(threshold=1000000, linewidth = 1000, precision = 5, sci_mode = False)

def attestation(t, m):
    for i in range(len(t)):
        for j in range(len(t)):
            assert t[i][j] == m[i][j]

def attest_v(t, m, tol = 4):
    for i in range(len(t)):
        try: assert round(t[i], tol) == round(m[i], tol); return
        except: pass
        #try: assert "{:.3f}".format(t[i]) == "{:.3f}".format(m[i])
        #except: print(t[i], m[i])

def copy(t): return [i for i in t]

def null(t): return [0 for _ in t]

def PageRankMatrix(event, fx, batch = False):
    edge_index  = event.edge_index
    edge_scores = event.edge_scores
    bin_top     = event.bin_top
    matrix      = event.bin_top_matrix
    mij         = event.Mij
    prg_i       = event.PR
    alpha = 0.85
    threshold = 0.5
    nux = 2 

    nodes = len(matrix)
    topx_mpx = [[-1 for _ in range(nodes)] for _ in range(nodes)]
    mx_top   = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for i in range(len(edge_index[0])):
        src, dst = edge_index[0][i], edge_index[1][i]
        binx = edge_scores[0][i] < edge_scores[1][i]

        try: assert binx == bin_top[i]
        except AssertionError: pass

        mx_top[src][dst] = edge_scores[1][i]
        if binx: topx_mpx[src][dst] = dst
    #print(torch.tensor(mx_top))

    attestation(matrix, mx_top)
    mx_ij = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for i in range(nodes):
        for j in range(nodes): mx_ij[i][j] = (i != j)*mx_top[i][j]
    attestation(mx_ij, mij)

    PR = [0 for _ in range(nodes)]
    for i in range(nodes):
        sm = 0
        for j in range(nodes): sm += mx_ij[j][i]
        sm = (1.0 / sm) if sm else 0
        for j in range(nodes): mx_ij[j][i] = (mx_ij[j][i]*sm  if sm else (1.0 / nodes))*alpha
        PR[i] = mx_top[i][i] / nodes
    attest_v(prg_i["0"], PR)
    #print(torch.tensor(mx_ij))

    xt = int(1)
    pr_x = copy(PR)  
    for k in range(xt):
        PR = null(PR)
        sx = 0

        for i in range(nodes):
            for j in range(nodes): PR[i] += mx_ij[i][j]*pr_x[j]
            PR[i] += (1 - alpha) / nodes
            sx += PR[i]


        norm = 0
        for i in range(nodes):
            PR[i] = PR[i] / sx
            norm += abs(PR[i] - pr_x[i])
            pr_x[i] = PR[i]

        #print(torch.tensor(pr_x), norm)
        try: attest_v(prg_i[str(k+1)], pr_x)
        except KeyError: pass

        #print(norm, k)
        if norm > 1e-6 and k < 1e6: continue
        #print(k, norm)

        norm = 0
        for i in range(nodes):
            sc = 0
            for j in range(nodes): sc += (i != j)*mx_ij[i][j]*PR[j]
            pr_x[i] = sc
            norm += sc

        if not norm: break
        for i in range(nodes): pr_x[i] = pr_x[i]/norm
        #for i in range(nodes): print("{:.3f}".format(pr_x[i]), "{:.3f}".format(prg_i["F"][i]))
        break

    if batch: return pr_x
    edge_index_t = torch.tensor(edge_index, device = "cuda")
    edge_score_t = torch.tensor(edge_scores, device = "cuda", dtype = torch.double)
    out = fx(edge_index_t, edge_score_t, alpha, 1e-6, threshold, int(xt))
    #print(nodes)
    t = torch.tensor(pr_x)
    c = out["pagerank"].view(-1).to(device = "cpu")
    ds = abs(t - c).sum(-1)
    #if ds > 0.1: print(ds); exit()
    return out["pagerank"]


    # This does the actual clustering 
    px = [-1 for _ in range(nodes)]
    ct = [[]  for _ in range(nodes)]
    for src in range(nodes):
        tmp = [-1 for _ in range(nodes)]
        ct[src] = tmp
        if not pr_x[src]: continue
        for n in topx_mpx[src]:
            if n == -1: continue
            if mx_top[src][n] < threshold: continue
            tmp[n] = n

            itx = iter([l for l in topx_mpx[n] if l > -1])
            for k in itx:
                if tmp[k] > -1 or topx_mpx[n][k] == -1: continue
                tmp[k] = k
                itx = iter([l for l in topx_mpx[k] if l > -1])

        lx = [f for f in tmp if f > -1]
        if len(lx) < nux: continue
        px[src] = sum([pr_x[f] for f in lx])
        ct[src] = sorted(tmp, reverse = True)

    print(torch.tensor(px, device = "cpu"))
    print(out["pagerank"].view(-1))

    print(torch.tensor(ct, device = "cpu"))
    print(out["nodes"])
    exit()
    return {"pagerank" : px, "nodes" : ct}


import time

#interpret(10000)
batch = []
data = loadsPage()
offno = 0
offset = []
pagerank = pyc().cupyc_graph_page_rank
mkx = []
xt = []
for i in data:
    #print(" ----------", i, "----------")
#    time.sleep(0.01)
    #axt = None
    #for _ in range(100000):
    #    t = PageRankMatrix(data[3])
    #    if axt is None: axt = t.clone(); continue
    #    if not torch.abs(t != axt).sum(-1): continue
    #    print(t)
    #    print(axt)
    #    exit()
    #pagerank = pyc()
    #edge_index_t = torch.tensor(data[i].edge_index, device = "cuda")
    #edge_score_t = torch.tensor(data[i].edge_scores, device = "cuda", dtype = torch.double)

    #axt = None
    #for _ in range(100000000):
    #    out = pagerank.cupyc_graph_page_rank(edge_index_t, edge_score_t, 0.85, 1e-6, 0.5, int(1e6))
    #    if axt is None: axt = {k : out[k].clone() for k in out}; continue
    #    same = torch.abs(axt["pagerank"] != out["pagerank"]).view(-1).sum(-1)
    #    if not same: continue
    #    for k in out:
    #        print("--------------------- " + k + " ---------------------")
    #        print(out[k])
    #        print(axt[k])
    #        print("Diff:")
    #        print(axt[k] == out[k])
    #    exit()
    #break
    #continue

#    if max(data[i].edge_index[0]) != 12: continue



    bx = 4
    if len(batch) >= bx: break
    evx = Event()
    batch.append(evx)
    mkx += [PageRankMatrix(data[i], pagerank, True)]
    xt += [PageRankMatrix(data[i], pagerank, False)]

    if len(batch) == 1: 
        evx = batch[0]
        evx.edge_index = [[],[]]
        evx.edge_scores = [[],[]]

    if len(batch): evx = batch[0]
    evx.edge_index[0] += [k for k in data[i].edge_index[0]]
    evx.edge_index[1] += [k for k in data[i].edge_index[1]]

    evx.edge_scores[0] += [k for k in data[i].edge_scores[0]]
    evx.edge_scores[1] += [k for k in data[i].edge_scores[1]]
    offset += [offno for k in range(len(data[i].edge_index[0]))]
    offno += data[i].num_nodes + 1

ev = batch[0]
ev.edge_index[0] = [offset[i] + ev.edge_index[0][i] for i in range(len(offset))]
ev.edge_index[1] = [offset[i] + ev.edge_index[1][i] for i in range(len(offset))]

mx = 0
for i in range(len(mkx)): mx = len(mkx[i]) if mx < len(mkx[i]) else mx
for i in range(len(mkx)): mkx[i] += [0] * (mx - len(mkx[i]))
print(mx)

edge_index_t = torch.tensor(ev.edge_index, device = "cuda")
edge_score_t = torch.tensor(ev.edge_scores, device = "cuda", dtype = torch.double)
out = pagerank(edge_index_t, edge_score_t, 0.85, 1e-6, 0.5, int(1))
print((out["remap"].view(-1, mx) > -1).sum(-1, True))

exit()
#exit()
#print(torch.tensor(mkx))
print(out["nodes"].to(device = "cpu").view(-1, mx))
exit()
print("_________")
for i in xt:
    print(i)



