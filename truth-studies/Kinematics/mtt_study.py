import itertools
import sys
sys.path.append('/nfs/dust/atlas/user/sitnikov/AnalysisTopGNN/models/NeutrinoReconstructionOriginal/')
from neutrino_momentum_reconstruction import doubleNeutrinoSolutions
import vector as v

def is_b_truthjet(truthjet):
    if sum([1 for parton in truthjet.Parton if abs(parton.pdgid) == 5]):
        return True
    return False

def get_b_truthjet_idx(event):
    result = [i for i, truthjet in enumerate(event.TruthJets) if truthjet.btagged and len(truthjet.Tops) != 0]
    return result

def get_b_truthjet_idx_var(event):
    result = [i for i, truthjet in enumerate(event.TruthJets) if is_b_truthjet(truthjet) and len(truthjet.Tops) != 0]
    return result

def is_one_b_per_top(event):
    for top in event.Tops:
        if sum([1 for truthjet in top.TruthJets if is_b_truthjet(truthjet)]) != 1:
            return False
    return True

def is_lep(child):
    return abs(child.pdgid) == 11 or abs(child.pdgid) == 13

def get_lep_child_idx(event):
    result = [i for i, child in enumerate(event.TopChildren) if is_lep(child)]
    return result

def is_nu(child):
    return abs(child.pdgid) == 12 or abs(child.pdgid) == 14

def get_nu_child_idx(event):
    result = [i for i, child in enumerate(event.TopChildren) if is_nu(child)]
    return result

def is_one_top_for_truthjet(event):
    if sum([1 for truthjet in event.TruthJets if len(truthjet.Tops) > 1]) != 0:
        return False
    return True


def is_event_ok(event):
    issues = []
    b_idx = get_b_truthjet_idx(event)
    count_lep_res = 0
    for top in event.Tops:
        if top.FromRes and sum([1 for child in top.Children if abs(child.pdgid) in [11, 13]]) != 0:
            count_lep_res += 1
    if count_lep_res != 1:
        issues.append(f'{count_lep_res}lepfromres')
    if len(event.Tops) != 4:
        issues.append('manytops')
    if len(b_idx) < 4:
        issues.append('not4b_partons_' + str(len(b_idx)))
    b_idx_var = get_b_truthjet_idx_var(event)
    if len(b_idx_var) < 4:
        issues.append('not4b_variable_' + str(len(b_idx_var)))
    if not is_one_b_per_top(event):
        issues.append('not1b1t')
    if not is_one_top_for_truthjet(event):
        issues.append('mergedjets')
    if len(get_lep_child_idx(event)) != 2:
        issues.append('not2lep')
    if sum([1 for child in event.TopChildren if abs(child.pdgid) == 15]) != 0:
        issues.append('tau')
    if len(issues) == 0:
        issues.append('ok')
    if len(issues) == 1 and 'partons' in issues[0]:
        issues.append('ok_variable')
    if len(issues) == 1 and 'variable' in issues[0]:
        issues.append('ok_partons')
    return issues

def get_jet_assignment_truth(event, jet_idx):
    result = []
    for idx in jet_idx:
        result.append(event.TruthJets[idx].TopIndex[0])
    return result

def get_lep_assignment_truth(event, lep_idx):
    result = []
    for idx in lep_idx:
        result.append(event.TopChildren[idx].TopIndex)
    return result

def get_nu_assignment_truth(event, nu_idx):
    result = []
    for idx in nu_idx:
        result.append(event.TopChildren[idx].TopIndex)
    return result

def get_had_lep_top_idx(event):
    result_had = []
    result_lep = []
    result_other = []
    for idx, top in enumerate(event.Tops):
        is_had = True
        is_lep = False
        for child in top.Children:
            if abs(child.pdgid) > 10:
                is_had = False
            if abs(child.pdgid) == 11 or abs(child.pdgid) == 13:
                is_lep = True
        if is_had and is_lep:
            print('BOTH HAD AND LEP')
        elif is_had:
            result_had.append(idx)
        elif is_lep:
            result_lep.append(idx)
        else:
            result_other.append(idx)
    return result_had, result_lep, result_other

def mtt_tops(event):
    res = []
    for top in event.Tops:
        if top.FromRes:
            res.append(top)
    return sum(res).CalculateMass()

def mtt_truthjets(event):
    res = []
    for truthjet in event.TruthJets:
        if len(truthjet.TopIndex) > 1:
            return None
        if truthjet.TopIndex[0] == -1:
            continue
        if event.Tops[truthjet.TopIndex[0]].FromRes:
            res.append(truthjet)
    for child in event.TopChildren:
        if 11 <= abs(child.pdgid) <= 14 and event.Tops[child.TopIndex].FromRes:
            res.append(child)
    return sum(res).CalculateMass()

def make_4vector(particle):
    return v.obj(e=particle.e, pt=particle.pt, eta=particle.eta, phi=particle.phi)

def reconstruct_mtt0(event):
    b_idx = get_b_truthjet_idx(event)
    jet_idx = [idx for idx in range(len(event.TruthJets)) if idx not in b_idx]
    lep_idx = get_lep_child_idx(event)
    nu_idx = get_nu_child_idx(event)

    tops = {i : [] for i in range(len(event.Tops))}
    for idx in b_idx:
        tops[event.TruthJets[idx].TopIndex[0]].append(event.TruthJets[idx])
    # for idx in tops:
    #     print(idx)
    jet_assignments = [get_jet_assignment_truth(event, jet_idx)]
    lep_assignments = [get_lep_assignment_truth(event, lep_idx)]

    for lep_assignment in lep_assignments:
        nu_assignment = get_nu_assignment_truth(event, nu_idx)
        for itop, ilep in zip(lep_assignment, lep_idx):
            # print(itop, ilep, lep_assignment, lep_idx)
            tops[itop].append(event.TopChildren[ilep])
        for itop, inu in zip(nu_assignment, nu_idx):
            tops[itop].append(event.TopChildren[inu])
        for jet_assignment in jet_assignments:
            for itop, ijet in zip(jet_assignment, jet_idx):
                if itop != -1:
                    tops[itop].append(event.TruthJets[ijet])
            # for itop in tops:
            #     print('top', itop)
            #     for obj in tops[itop]:
            #         print(obj)
            res_products = []
            for itop in tops:
                if itop < len(event.Tops) and event.Tops[itop].FromRes:
                    res_products += tops[itop]
    print('-----------------')
    for obj in res_products:
        print(obj)
    if len(res_products) == 0:
        return None
    print(sum(res_products).CalculateMass())
    res_vect = v.obj(x=0, y=0, z=0, t=0)
    for prod in res_products:
        res_vect += make_4vector(prod)
    print(res_vect.tau)
    return sum(res_products).CalculateMass()

def select_resonance(tops, lep_tops):
    res_products = []
    for i in tops:
        tops[i] = sum(tops[i])
    if tops[lep_tops[0]].pt > tops[lep_tops[1]].pt:
        res_products.append(tops[lep_tops[0]])
    else:
        res_products.append(tops[lep_tops[1]])
    had_tops = [itop for itop in tops if itop not in lep_tops]
    if tops[had_tops[0]].pt > tops[had_tops[1]].pt:
        res_products.append(tops[had_tops[0]])
    else:
        res_products.append(tops[had_tops[1]])
    return res_products

def reconstruct_mtt1(event):
    b_idx = get_b_truthjet_idx(event)
    jet_idx = [idx for idx in range(len(event.TruthJets)) if idx not in b_idx]
    lep_idx = get_lep_child_idx(event)
    nu_idx = get_nu_child_idx(event)

    tops = {i : [] for i in range(len(event.Tops))}
    for idx in b_idx:
        tops[event.TruthJets[idx].TopIndex[0]].append(event.TruthJets[idx])
    # for idx in tops:
    #     print(idx)
    jet_assignments = [get_jet_assignment_truth(event, jet_idx)]
    lep_assignments = [get_lep_assignment_truth(event, lep_idx)]

    for lep_assignment in lep_assignments:
        nu_assignment = get_nu_assignment_truth(event, nu_idx)
        for itop, ilep in zip(lep_assignment, lep_idx):
            # print(itop, ilep, lep_assignment, lep_idx)
            tops[itop].append(event.TopChildren[ilep])
        for itop, inu in zip(nu_assignment, nu_idx):
            tops[itop].append(event.TopChildren[inu])
        for jet_assignment in jet_assignments:
            for itop, ijet in zip(jet_assignment, jet_idx):
                if itop != -1:
                    tops[itop].append(event.TruthJets[ijet])
    res_products = select_resonance(tops, lep_assignment)
    if len(res_products) == 0:
        return None
    return sum(res_products).CalculateMass()

def find_diff(v1, v2):
    return ((v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2)**0.5

def select_result_using_nu_truth(nu_4vec_truth, result):
    result_3vec = []
    def make_vector(obj):
        return v.obj(x=obj[0], y=obj[1], z=obj[2])
    for lv in result:
        result_3vec.append((make_vector(lv[0]), make_vector(lv[1])))
    tv1 = nu_4vec_truth[0]
    tv2 = nu_4vec_truth[1]
    answer = {}
    for j in range(len(result_3vec)):
        rv1 = result_3vec[j][0]
        rv2 = result_3vec[j][1]
        c11 = find_diff(tv1, rv1)
        c22 = find_diff(tv2, rv2)
        c21 = find_diff(tv2, rv1)
        c12 = find_diff(tv1, rv2)
        answer[f'{j} same'] = c11 + c22
    choice = min(answer, key=answer.get)
    number_reco = int(choice.split(' ')[0])
    nu1_res = result_3vec[number_reco][0]
    nu1 = v.obj(x=nu1_res.x, y=nu1_res.y, z=nu1_res.z, m=0)
    nu2_res = result_3vec[number_reco][1]
    nu2 = v.obj(x=nu2_res.x, y=nu2_res.y, z=nu2_res.z, m=0)
    if 'opposite' in choice:
        nu1, nu2 = nu2, nu1
    return nu1, nu2

def get_dR(v1, v2):
    return((v1.eta - v2.eta)**2 + (v1.phi - v2.phi)**2)**0.5

def reconstruct_mtt2(event):
    b_idx = get_b_truthjet_idx(event)
    jet_idx = [idx for idx in range(len(event.TruthJets)) if idx not in b_idx]
    lep_idx = get_lep_child_idx(event)

    tops = {i : [] for i in range(len(event.Tops))}
    for idx in b_idx:
        tops[event.TruthJets[idx].TopIndex[0]].append(event.TruthJets[idx])
    # for idx in tops:
    #     print(idx)
    jet_assignments = [get_jet_assignment_truth(event, jet_idx)]
    lep_assignments = [get_lep_assignment_truth(event, lep_idx)]

    for lep_assignment in lep_assignments:
        nu_assignment = lep_assignment
        for itop, ilep in zip(lep_assignment, lep_idx):
            # print(itop, ilep, lep_assignment, lep_idx)
            tops[itop].append(event.TopChildren[ilep])
        # try:
        #     bs = [make_4vector(ebemt.TruthJets[idx]) for idx in b_idx if]
        #     nu_solutions = NMR.doubleNeutrinoSolutions(bs[0], bs[1], leps[0], leps[1], event.met*np.cos(event.met_phi), event.met*np.sin(event.met_phi))
        #     nu_res = select_result_using_nu_truth(nus, nu_solutions.nunu_s)
        #     for top_type in ['res', 'spec']:
        #         if event_type + '_' + top_type not in diffs:
        #             diffs[event_type + '_' + top_type] = []
        #     for k in range(2):
        #         # print('nu reco:', nu_res[k].x, nu_res[k].y, nu_res[k].z, nu_res[k].eta, nu_res[k].phi)
        #         # print('nu truth:', nus[k].x, nus[k].y, nus[k].z, nus[k].eta, nus[k].phi)
        #         if from_res[k]:
        #             diffs[f'{event_type}_res'].append(get_dR(nu_res[k], nus[k]))
        #         else:
        #             diffs[f'{event_type}_spec'].append(get_dR(nu_res[k], nus[k]))
        #     # break
        # except:
        #     continue
        for itop, inu in zip(nu_assignment, nu_idx):
            tops[itop].append(event.TopChildren[inu])
        for jet_assignment in jet_assignments:
            for itop, ijet in zip(jet_assignment, jet_idx):
                if itop != -1:
                    tops[itop].append(event.TruthJets[ijet])
    res_products = select_resonance(tops, lep_assignment)
    if len(res_products) == 0:
        return None
    return sum(res_products).CalculateMass()
