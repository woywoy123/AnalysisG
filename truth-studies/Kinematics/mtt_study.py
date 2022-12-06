import itertools
import sys
sys.path.append('/nfs/dust/atlas/user/sitnikov/ntuples_for_classifier/')
# from neutrino_momentum_reconstruction_python3 import doubleNeutrinoSolutions

def is_b_truthjet(truthjet):
    if sum([1 for parton in truthjet.Partons if abs(parton.pdgid) == 5]):
        return True
    return False

def is_b_truthjet_parent(truthjet):
    if sum([1 for child in truthjet.Parent if abs(child.pdgid) == 5]):
        return True
    return False

def get_b_truthjet_idx(event):
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
    nbjparent = len([1 for truthjet in event.TruthJets if is_b_truthjet_parent(truthjet)])
    if nbjparent < 4:
        issues.append('not4b_parent_' + str(nbjparent))
    b_idx = get_b_truthjet_idx(event)
    if len(event.Tops) != 4:
        issues.append('manytops')
    if len(b_idx) < 4:
        issues.append('not4b_partons_' + str(len(b_idx)))
    if not is_one_b_per_top(event):
        issues.append('not1b1t')
    if not is_one_top_for_truthjet(event):
        issues.append('mergedjets')
    if len(get_lep_child_idx(event)) != 2:
        issues.append('not2lep')
    if len(issues) == 0:
        issues.append('ok_both')
    if len(issues) == 1 and 'parent' in issues[0]:
        issues.append('ok_partons')
    if len(issues) == 1 and ['partons'] in issues[0]:
        issues.append('ok_parent')
    return issues

def get_jet_assignment_truth(event, jet_idx):
    result = []
    for idx in jet_idx:
        result.append(event.TruthJets[idx].index[0])
    return result

def get_lep_assignment_truth(event, lep_idx):
    result = []
    for idx in lep_idx:
        result.append(event.TopChildren[idx].index)
    return result

def get_nu_assignment_truth(event, nu_idx):
    result = []
    for idx in nu_idx:
        result.append(event.TopChildren[idx].index)
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
        if len(truthjet.index) > 1:
            return None
        if truthjet.index[0] == -1:
            continue
        if event.Tops[truthjet.index[0]].FromRes:
            res.append(truthjet)
    for child in event.TopChildren:
        if 11 <= abs(child.pdgid) <= 14 and event.Tops[child.index].FromRes:
            res.append(child)
    return sum(res).CalculateMass()


def reconstruct_mtt(event):
    b_idx = get_b_truthjet_idx(event)
    jet_idx = [idx for idx in range(len(event.TruthJets)) if idx not in b_idx]
    lep_idx = get_lep_child_idx(event)
    nu_idx = get_nu_child_idx(event)

    tops = {i : [] for i in range(6)}
    for idx in b_idx:
        tops[event.TruthJets[idx].index[0]].append(event.TruthJets[idx])
    # for idx in tops:
    #     print(idx)
    jet_assignments = [get_jet_assignment_truth(event, jet_idx)]
    lep_assignments = [get_lep_assignment_truth(event, lep_idx)]
    top_had_idx, top_lep_idx, top_other_idx = get_had_lep_top_idx(event)

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
            res_products = []
            for itop in tops:
                if itop < len(event.Tops) and event.Tops[itop].FromRes:
                    res_products += tops[itop]
    if len(res_products) == 0:
        return None
    return sum(res_products).CalculateMass()
