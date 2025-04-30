# distutils: language=c++
# cython: language_level=3

from cython.operator cimport dereference as dref
from AnalysisG.core.tools cimport *
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport int, bool

cdef extern from "<tools/merge_cast.h>" nogil:
    cdef void merge_data(vector[particle_template*]* out, vector[particle_template*]* p2)
    cdef void merge_data(vector[neutrino*         ]* out, vector[neutrino*         ]* p2)
    cdef void release_vector(vector[atomic]*)

    cdef void release_vector(vector[vector[double]]*)
    cdef void release_vector(vector[double]*)
    cdef void release_vector(vector[int]*)
    cdef void release_vector(vector[particle_template*]*)
    cdef void release_vector(vector[neutrino*]*)




cdef void assign_particles(string name, vector[particle_template*]* ptx, event_t* ev):
    cdef int ix
    for ix in range(ptx.size()): ev.truth_tops[name][ix%2].push_back(ptx.at(ix))
    release_vector(&ev.truth_tops[name][0])
    release_vector(&ev.truth_tops[name][1])

cdef void assign_neutrinos(string name, vector[neutrino*]* pt1, vector[neutrino*]* pt2, event_t* ev):
    cdef int ix
    for ix in range(pt1.size()): 
        ev.reco_tops[name][0].push_back(pt1.at(ix))
        ev.reco_tops[name][1].push_back(pt2.at(ix))
    release_vector(&ev.reco_tops[name][0])
    release_vector(&ev.reco_tops[name][1])

cdef vector[particle_template*] make_particle(vector[vector[double]]* pmu, vector[int]* pdgid):
    cdef int ix 
    cdef particle_template* ptx = NULL
    cdef vector[particle_template*] out
    for ix in range(pmu.size()):
        ptx = new particle_template()
        ptx.pt  = pmu.at(ix).at(0); ptx.eta = pmu.at(ix).at(1)
        ptx.phi = pmu.at(ix).at(2); ptx.e   = pmu.at(ix).at(3)
        ptx.pdgid = pdgid.at(ix)
        out.push_back(ptx)
    pmu.clear()
    pdgid.clear()
    release_vector(pmu)
    release_vector(pdgid)
    return out

cdef vector[neutrino*] make_neutrino(vector[vector[double]]* pmu, vector[int]* lep, vector[int]* bq, vector[double]* elp, vector[double]* chi):
    cdef int ix 
    cdef neutrino* ptx = NULL
    cdef vector[neutrino*] out
    for ix in range(pmu.size()):
        ptx = new neutrino()
        ptx.type = string(b"neutrino")
        ptx.pt  = pmu.at(ix).at(0); ptx.eta = pmu.at(ix).at(1)
        ptx.phi = pmu.at(ix).at(2); ptx.e   = pmu.at(ix).at(3)
        ptx.matched_bquark = bq.at(ix); ptx.matched_lepton = lep.at(ix)
        ptx.ellipse = elp.at(ix); ptx.chi2 = chi.at(ix); 
        out.push_back(ptx)
    pmu.clear(); release_vector(pmu)
    chi.clear(); release_vector(chi)
    return out


cdef atomic* get_ref(vector[atomic]* conx):
    cdef int ix = conx.size()
    cdef map[int, sets] cm
    conx.push_back(atomic(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cm, cm, cm, cm, cm, b""))
    return &conx.at(ix)

cdef particle_template* protect(vector[particle_template*]* inx, bool nu, bool b, bool lep):
    if inx == NULL: return NULL

    cdef int ix
    cdef particle_template* ptx
    for ix in range(inx.size()):
        ptx = inx.at(ix)
        if   b   and <bool>(ptx.is_b):   return ptx
        elif nu  and <bool>(ptx.is_nu):  return ptx
        elif lep and <bool>(ptx.is_lep): return ptx
    return NULL

cdef string bar(int p, string sym): return sym if p > 0 else b"\\bar{" + sym + b"}"

cdef string _pdgid(int p):
    if abs(p) == 11: return bar(p, b"e")
    if abs(p) == 12: return bar(p, b"\\nu_{e}")
    if abs(p) == 13: return bar(p, b"\\mu")
    if abs(p) == 14: return bar(p, b"\\nu_{\\mu}")
    if abs(p) == 15: return bar(p, b"\\tau")
    if abs(p) == 16: return bar(p, b"\\nu_{\\tau}")
    print("!!!!!!!!!", p); exit()
    return b""

cdef string get_pdg(vector[particle_template*] p):
    cdef map[int, vector[particle_template*]] dx
    cdef particle_template* i
    cdef vector[particle_template*] xt
    for i in p:
        if i == NULL: continue
        xt.push_back(i)
    if not xt.size(): return b"None"

    cdef int pdg
    cdef vector[int] dxp

    cdef string kx = b""
    for i in xt:
        pdg = abs(i.pdgid)
        dx[pdg].push_back(i)
        dxp.push_back(pdg)
    for pdg in sorted(list(set(dxp))): 
        for i in dx[pdg]: kx += _pdgid(i.pdgid)
    return b"(" + kx + b")"

cdef sets make_sets(
        particle_template* tq, particle_template* tl,
        particle_template* tn, neutrino* rn, 
        selection_template* slx, string* sym
):
    cdef sets sx = sets(tq, tl, tn, rn, NULL, NULL, NULL, NULL, rn.ellipse, rn.chi2, b"")

    cdef vector[particle_template*] vc
    vc.push_back(tn)
    vc.push_back(tl)
    sx.symbolic = get_pdg(vc)

    slx.sum(&vc, &sx.tru_wboson)
    vc.push_back(tq)
    slx.sum(&vc, &sx.tru_top)
    vc.clear()

    vc.push_back(<particle_template*>(rn))
    vc.push_back(tl)
    slx.sum(&vc, &sx.rec_wboson)
    
    vc.push_back(tq)
    slx.sum(&vc, &sx.rec_top)
    sym.append(sx.symbolic)
    return sx


cdef void match(atomic* atm, map[int, vector[particle_template*]]* ts, map[int, vector[particle_template*]]* rs, bool chix, selection_template* slx):
    cdef vector[particle_template*]* nt1 = &ts.at(0) if ts.at(0).size() else NULL
    cdef vector[particle_template*]* nt2 = &ts.at(1) if ts.at(1).size() else NULL
    cdef vector[particle_template*]* nr1 = &rs.at(0) if rs.at(0).size() else NULL
    cdef vector[particle_template*]* nr2 = &rs.at(1) if rs.at(1).size() else NULL

    # -------- count the neutrinos ------- #
    atm.n_tru_nu += 0 if nt1 == NULL else 1
    atm.n_tru_nu += 0 if nt2 == NULL else 1
    atm.n_rec_nu += 0 if nr1 == NULL else 1
    atm.n_rec_nu += 0 if nr2 == NULL else 1
    atm.num_sols += 0 if nr1 == NULL else nr1.size()
    if not atm.n_rec_nu: atm.symbolics = b"no-solutions"; return # all nus as lost if self.n_tru_nu > 0

    cdef particle_template* tru_nu1 = protect(nt1, True , False, False)
    cdef particle_template* tru_bq1 = protect(nt1, False,  True, False)
    cdef particle_template* tru_lp1 = protect(nt1, False, False,  True)

    cdef particle_template* tru_nu2 = protect(nt2, True , False, False)
    cdef particle_template* tru_bq2 = protect(nt2, False,  True, False)
    cdef particle_template* tru_lp2 = protect(nt2, False, False,  True)

    atm.merged_jet += 0 if tru_bq1 == NULL or tru_bq2 == NULL else string(tru_bq1.hash) == string(tru_bq2.hash)
    # -------- check whether the algorithm triggered in a non-dilepton event ------- #
    cdef tools tl = tools() 

    cdef neutrino* nu1 = NULL
    cdef neutrino* nu2 = NULL
    cdef particle_template* ptx

    cdef int ix
    cdef double f = (1.0 / 1000000)
    cdef double ch2 = 0
    cdef vector[double] ch2v
    cdef map[double, vector[neutrino*]] chx
    for ix in range(nr1.size()):
        ptx = nr1.at(ix)
        nu1 = <neutrino*>(ptx)
        nu1.chi2 = nu1.chi2 * (1*(nu1.chi2 < 0) + f*(nu1.chi2 >= 0))

        ptx = nr2.at(ix)
        nu2 = <neutrino*>(ptx)
        nu2.chi2 = nu2.chi2 * (1*(nu2.chi2 < 0) + f*(nu2.chi2 >= 0))

        ch2 = (nu1.chi2 + nu2.chi2)
        chx[ch2].push_back(nu1)
        chx[ch2].push_back(nu2)
        ch2v.push_back(ch2)
    ch2v = <vector[double]>(sorted(ch2v))
    ch2 = dref(ch2v.begin()); ch2v.erase(ch2v.begin())
    if ch2 < 0: atm.n_non_nunu += 2
    if not chix: nu1, nu2 = <neutrino*>(dref(nr1)[0]), <neutrino*>(dref(nr2)[0])
    elif chix and ch2 < 0 and not ch2v.size(): nu1, nu2 = <neutrino*>(NULL), <neutrino*>(NULL)
    elif chix and ch2 < 0 and ch2v.size(): nu1, nu2 = chx[ch2v[0]][0], chx[ch2v[0]][1]
    else: nu1, nu2 = chx[ch2][0], chx[ch2][1]
    if nu1 == NULL or nu2 == NULL: atm.symbolics = b"fakes"; return 

    # ------ matched ----- #
    cdef bool mb = nu1.matched_bquark == 1 and nu2.matched_bquark == 1
    cdef bool ml = nu1.matched_lepton == 1 and nu2.matched_lepton == 1

    # ------ swapped ----- #
    cdef bool sb = nu1.matched_bquark == -1 and nu2.matched_bquark == -1
    cdef bool sl = nu1.matched_lepton == -1 and nu2.matched_lepton == -1

    # ------ perfect match ----- #
    if mb and ml: 
        atm.n_correct += 2
        atm.correct[0] = make_sets(tru_bq1, tru_lp1, tru_nu1, nu1, slx, &atm.symbolics)
        atm.correct[1] = make_sets(tru_bq2, tru_lp2, tru_nu2, nu2, slx, &atm.symbolics)

    # ------ swapped lepton and bquarks ----- #
    elif sb and sl: 
        atm.n_l_swapped += 1
        atm.n_b_swapped += 1
        atm.swapped_bl[0] = make_sets(tru_bq1, tru_lp1, tru_nu1, nu1, slx, &atm.symbolics)
        atm.swapped_bl[1] = make_sets(tru_bq2, tru_lp2, tru_nu2, nu2, slx, &atm.symbolics)

    # ------ swapped bquarks but matching leptons ------ # 
    elif sb and ml: 
        atm.n_b_swapped += 2
        atm.swapped_bs[0] = make_sets(tru_bq1, tru_lp1, tru_nu1, nu1, slx, &atm.symbolics)
        atm.swapped_bs[1] = make_sets(tru_bq2, tru_lp2, tru_nu2, nu2, slx, &atm.symbolics)

    # ------ swapped leptons but matching bquarks ------- #
    elif sl and mb: 
        atm.n_l_swapped += 2
        atm.swapped_ls[0] = make_sets(tru_bq1, tru_lp1, tru_nu1, nu1, slx, &atm.symbolics)
        atm.swapped_ls[1] = make_sets(tru_bq2, tru_lp2, tru_nu2, nu2, slx, &atm.symbolics)

    # ----------------- fake neutrino ----------------- #
    else: 
        atm.n_unmatched += 2
        atm.fake_nus[0] = make_sets(tru_bq1, tru_lp1, tru_nu1, nu1, slx, &atm.symbolics)
        atm.fake_nus[1] = make_sets(tru_bq2, tru_lp2, tru_nu2, nu2, slx, &atm.symbolics)

cdef void add_container(string name, container* con, event_t* ev, bool ch2):
    cdef atomic* at 
    if not ch2: at = get_ref(&con.atomics[name])
    else:  at = get_ref(&con.chi2_atomics[name])
    match(at, &ev.truth_tops[name], &ev.reco_tops[name], ch2, con.slx)

cdef export_t get_export(string name, container* con, bool ch2):
    cdef vector[atomic]* datax = &con.atomics[name] if ch2 else &con.chi2_atomics[name]
    cdef export_t exp
    cdef atomic* atm
    cdef int ix, i

    for ix in range(datax.size()):
        atm = &datax.at(ix)

        exp.n_correct[atm.symbolics].push_back(atm.n_correct)
        exp.n_b_swapped[atm.symbolics].push_back(atm.n_b_swapped)
        exp.n_l_swapped[atm.symbolics].push_back(atm.n_l_swapped)
        exp.n_bl_swapped[atm.symbolics].push_back(atm.n_bl_swapped)
        exp.n_unmatched[atm.symbolics].push_back(atm.n_unmatched)
        exp.n_non_nunu[atm.symbolics].push_back(atm.n_non_nunu)
        exp.n_tru_nu[atm.symbolics].push_back(atm.n_tru_nu)
        exp.n_rec_nu[atm.symbolics].push_back(atm.n_rec_nu)
        exp.num_sols[atm.symbolics].push_back(atm.num_sols)
        exp.merged_jet[atm.symbolics].push_back(atm.merged_jet)

        for i in range(2):
            exp.correct[i][atm.symbolics].push_back(atm.correct[i])
            exp.swapped_bs[i][atm.symbolics].push_back(atm.swapped_bs[i])
            exp.swapped_bl[i][atm.symbolics].push_back(atm.swapped_bl[i])
            exp.swapped_ls[i][atm.symbolics].push_back(atm.swapped_ls[i])
            exp.fake_nus[i][atm.symbolics].push_back(atm.fake_nus[i])

    datax.clear()
    release_vector(datax)
    return exp
