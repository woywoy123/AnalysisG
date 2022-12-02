import uproot
import ROOT
from tqdm import tqdm
import plotly.graph_objects as go

def get_data_from_file_new_format(file_name):
    f = uproot.open(file_name)
    n = f['nominal']
    var_names = ['children_pt',
                 'children_e',
                 'children_eta',
                 'children_phi',
                 'children_pdgid',
                 'top_FromRes',
                 'met_met',
                 'met_phi',
                 'truthjet_TopIndex',
                 'top_pt', 'top_e', 'top_eta', 'top_phi', 'top_charge',
                 'truJparton_pt',
                 'truJparton_e',
                 'truJparton_eta',
                 'truJparton_phi',
                 'truJparton_charge',
                 'truJparton_pdgid',
                 'truJparton_ChildIndex',
                 'truJparton_TruJetIndex',
                ]
    for obj_type in ['truthjet']:
        var_names += [f'{obj_type}_{var_type}' for var_type in ['pt', 'e', 'eta', 'phi']]
        if obj_type in ['el', 'mu']:
            var_names.append(f'{obj_type}_charge')
            var_names.append(f'{obj_type}_true_origin')
        if obj_type == 'jet':
            var_names.append(f'{obj_type}_truthflav')
    data = n.arrays(var_names, library='np')
    nentries = len(list(data.values())[0])
    return data, nentries


loss = {num : 0 for num in range(0, 7)}
top_count = {'accept' : 0, 'lep' : 0, 'had' : 0, 'lepj' : 0}
total_top_count = [0]


def is_entry_ok(data, ientry, data_type):
    truthjet_TopIndex = data['truthjet_TopIndex'][ientry]
    is_ok = True
    if len(data['top_pt'][ientry]) != 4:
        loss[0] += 1
        is_ok = False
    lep_tops = []
    had_tops = []
    top_from_res = []
    for top_idx in range(len(data['top_pt'][ientry])):
        total_top_count[0] += 1
        if int(data['top_FromRes'][ientry][top_idx]):
            top_from_res.append(top_idx)
        is_lep = False
        is_had = True
        for child_idx in range(len(data['children_pt'][ientry][top_idx])):
            if abs(data['children_pdgid'][ientry][top_idx][child_idx]) == 11 or abs(data['children_pdgid'][ientry][top_idx][child_idx]) == 13:
                is_lep = True
            if abs(data['children_pdgid'][ientry][top_idx][child_idx]) > 10:
                is_had = False
        if is_lep:
            lep_tops.append(top_idx)
            top_count['accept'] += 1
            top_count['lep'] += 1
        elif is_had:
            had_tops.append(top_idx)
            top_count['accept'] += 1
        else:
            loss[1] += 1
            is_ok = False
    if len(lep_tops) != 2 or len(had_tops) != 2:
        loss[2] += 1
        is_ok = False
    if data_type != 'SM':
        if len(top_from_res) != 2 or (top_from_res[0] in lep_tops and top_from_res[1] in lep_tops):
            loss[3] += 1
            is_ok = False
        if len(top_from_res) != 2 or (top_from_res[0] in had_tops and top_from_res[1] in had_tops):
            loss[3] += 1
            is_ok = False
    c4_ok = True
    for entry in truthjet_TopIndex:
        if len(entry) > 1:
            c4_ok = False
    if not c4_ok:
        loss[4] += 1
        is_ok = False
    jets_to_tops = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0}
    for truthjet_idx in range(len(data['truthjet_pt'][ientry])):
        if truthjet_TopIndex[truthjet_idx][0] == -1:
            continue
        jets_to_tops[truthjet_TopIndex[truthjet_idx][0]] += 1
    c5_ok = True
    for top_idx in lep_tops:
        if jets_to_tops[top_idx] != 1:
            c5_ok = False
        else:
            top_count['lepj'] += 1
    if not c5_ok:
        loss[5] += 1
        is_ok = False
    c6_ok = True
    for top_idx in had_tops:
        if jets_to_tops[top_idx] < 1:
            c6_ok = False
        if jets_to_tops[top_idx] == 3:
            top_count['had'] += 1
    if not c6_ok:
        loss[6] += 1
        is_ok = False
    return is_ok

def make_4vector(e, pt, eta, phi):
    result = ROOT.TLorentzVector()
    result.SetPtEtaPhiE(pt, eta, phi, e)
    return result

def make_plot(values, name, plotting_range=None, nbins=200, labels=['SM', 'BSM_400', 'BSM_1000'], x_label="", y_label="", title=""):
#     median_value = np.median(values['SM'])
    if plotting_range == None:
        values_restricted = values
    else:
        values_restricted = {label : [x for x in values[label] if x >= plotting_range[0] and x <= plotting_range[1]] for label in labels}
#     print(values_restricted)
    fig = go.Figure()
    fig.add_traces([
        go.Histogram(x=values_restricted[label], name=label,
                     histnorm='percent', nbinsx=nbins, opacity=0.4)
                    for label in labels])
    fig.update_layout({'barmode' : 'overlay', "xaxis_title" : x_label, "yaxis_title" : y_label, "title" : title})
#     fig.show()
    fig.write_image(f'plots_root/{name}.png')


def reconstruct_mtt(data, ientry):
    top_4vecs = [ROOT.TLorentzVector() for i in range(4)]
    res_4vec = ROOT.TLorentzVector()
    from_res = [int(i) for i in data['top_FromRes'][ientry]]
    for truthjet_idx in range(len(data['truthjet_pt'][ientry])):
        top_idx = data['truthjet_TopIndex'][ientry][truthjet_idx][0]
        if top_idx != -1 and from_res[top_idx]:
            res_4vec += make_4vector(data['truthjet_e'][ientry][truthjet_idx],
                                                   data['truthjet_pt'][ientry][truthjet_idx],
                                                   data['truthjet_eta'][ientry][truthjet_idx],
                                                   data['truthjet_phi'][ientry][truthjet_idx]
                                                  )
    for top_idx in range(4):
        if not from_res[top_idx]:
            continue
        for child_idx in range(len(data['children_pt'][ientry][top_idx])):
            if 11 <= abs(data['children_pdgid'][ientry][top_idx][child_idx]) <= 14:
                res_4vec += make_4vector(data['children_e'][ientry][top_idx][child_idx],
                                                       data['children_pt'][ientry][top_idx][child_idx],
                                                       data['children_eta'][ientry][top_idx][child_idx],
                                                       data['children_phi'][ientry][top_idx][child_idx]
                                                      )
                return res_4vec.M()
    return False




import os
file_paths = []
base_directory = '/nfs/dust/atlas/user/sitnikov/ntuples_for_classifier'
directories = os.listdir(f'{base_directory}')
for directory in directories:
    if ('user.esitniko' in directory and 'job06' in directory) or \
       'user.tnommens' in directory:
        print('==================')
        print(directory)
        files = os.listdir(f'{base_directory}/{directory}')
        for file_name in files:
            print(file_name.split('_')[-1].split('.')[0])
            file_paths.append(f'{base_directory}/{directory}/{file_name}')


datas = []
count_sm = 0
ccc = 0
for file_path in file_paths:
    print(file_path)
    data_type = 'BSM_400' if '312440' in file_path else 'BSM_1000' if '312446' in file_path else 'SM'
    if data_type == 'SM' and count_sm >= 0:
        continue
    if data_type == 'SM':
        count_sm += 1
    if data_type == 'BSM_400':
        continue

    for array in uproot.iterate(f'{file_path}:nominal', ['truthjet_pt'], step_size=100, library='np'):
        for chunk in array['truthjet_pt']:
        # print(array['truthjet_pt'], len(array['truthjet_pt']))
            ccc += len(chunk)
    data, nentries = get_data_from_file_new_format(file_path)
    datas.append({'type' : data_type, 'data' : data, 'nentries' : nentries})


print(ccc)



n_truthjets = 0


mtts = {'SM' : [], 'BSM_400' : [], 'BSM_1000' : []}
topmasses = {'SM' : [], 'BSM_400' : [], 'BSM_1000' : []}
mtt_top = {'SM' : [], 'BSM_400' : [], 'BSM_1000' : []}
count = 0
for data_item in datas:
    nentries = data_item['nentries']
    data = data_item['data']
    data_type = data_item['type']
    for ientry in tqdm(range(nentries)):
        n_truthjets += len(data['truthjet_pt'][ientry])
        count += 1
        if is_entry_ok(data, ientry, data_type):
            mtt = reconstruct_mtt(data, ientry)
            mtts[data_type].append(mtt/1000)
        tops = [[] for i in range(len(data['top_pt'][ientry]))]
        tops_ok = [True for i in range(len(tops))]
        top_type = [None for i in range(len(tops))]
        for top_idx in range(len(data['top_pt'][ientry])):
            is_had = True
            is_lep = False
            for child_idx in range(len(data['children_pt'][ientry][top_idx])):
                if abs(data['children_pdgid'][ientry][top_idx][child_idx]) > 10:
                    is_had = False
                if 11 <= abs(data['children_pdgid'][ientry][top_idx][child_idx]) >= 14:
                    is_lep = True
            if is_had:
                top_type[top_idx] = 'Had'
            if is_lep:
                top_type[top_idx] = 'Lep'
        for truthjet_idx in range(len(data['truthjet_pt'][ientry])):
            top_idx = data['truthjet_TopIndex'][ientry][truthjet_idx]
            if len(top_idx) != 1:
                for idx in top_idx:
                    tops_ok[idx] = False
            elif top_idx[0] != -1:
                tops[top_idx[0]].append( make_4vector(data['truthjet_e'][ientry][truthjet_idx],
                                                       data['truthjet_pt'][ientry][truthjet_idx],
                                                       data['truthjet_eta'][ientry][truthjet_idx],
                                                       data['truthjet_phi'][ientry][truthjet_idx]
                                                      ))
        for top_idx in range(len(data['children_pt'][ientry])):
            for child_idx in range(len(data['children_pt'][ientry][top_idx])):
                if 11 <= abs(data['children_pdgid'][ientry][top_idx][child_idx]) <= 14:
                     tops[top_idx].append( make_4vector(data['children_e'][ientry][top_idx][child_idx],
                                                            data['children_pt'][ientry][top_idx][child_idx],
                                                            data['children_eta'][ientry][top_idx][child_idx],
                                                            data['children_phi'][ientry][top_idx][child_idx]
                                                           ))
        for top_idx in range(len(top_type)):
            if tops_ok[top_idx] and top_type[top_idx] in ['Had', 'Lep'] and len(tops[top_idx]) == 3:
                top = ROOT.TLorentzVector()
                for child in tops[top_idx]:
                    top += child
                topmasses[data_type].append(top.M()/1000)

        res_tops = []
        for top_idx in range(len(top_type)):
            if tops_ok[top_idx] and top_type[top_idx] in ['Had', 'Lep'] and int(data['top_FromRes'][ientry][top_idx]):
                res_tops.append(tops[top_idx])
        if len(res_tops) == 2:
            top4vec = ROOT.TLorentzVector()
            for top in res_tops:
                for child in top:
                    top4vec += child
            mtt_top[data_type].append(top4vec.M()/1000)


make_plot(mtts, 'truthjet_mtt/mtt_test', plotting_range=(0, 1500), nbins=200)
make_plot(topmasses, 'truthjet_mtt/topmass', plotting_range=(100, 300), nbins=200)
make_plot(mtt_top, 'truthjet_mtt/mtt_tops', plotting_range=(0, 1500), nbins=200)

print(len(mtts))
print(len(topmasses))
print(len(mtt_top))


print(loss)
print(count)
print({k : loss[k]/count for k in loss})

print(top_count)
print(total_top_count)

print(f'{n_truthjets} truthjets')
