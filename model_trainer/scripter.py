import yaml
import pathlib

samples = [
#    ("sm_tttt"        , -1),
#    ("sm_ttt"         , -1),
#    ("bsm_ttttH_400"  , -1),
#    ("bsm_ttttH_500"  , -1),
#    ("bsm_ttttH_600"  , -1),
#    ("bsm_ttttH_700"  , -1),
#    ("bsm_ttttH_800"  , -1),
#    ("bsm_ttttH_900"  , -1),
#    ("bsm_ttttH_1000" , -1),
#    ("sm_ttbar"       , -1),
#    ("sm_ttV"         , -1),
#    ("sm_tt_Vll"      , -1),
#    ("sm_Vll"         , -1),
#    ("sm_llgammagamma", -1),
#    ("sm_ttH"         , -1),
#    ("sm_t"           , -1),
#    ("sm_wh"          , -1),
#    ("sm_VVll"        , -1),
#    ("sm_llll"        , -1),
    ("/home/tnom6927/Downloads/test/*", -1)
]

params = [
    ("MRK-1", "adam", {"lr" : 1e-6}                                       , "cuda:0"),
#    ("MRK-2", "adam", {"lr" : 1e-8}                                       , "cuda:0"),
#    ("MRK-3", "adam", {"lr" : 1e-8, "amsgrad" : True}                     , "cuda:0"),
#    ("MRK-4", "sgd",  {"lr" : 1e-6}                                       , "cuda:0"),
#    ("MRK-5", "sgd",  {"lr" : 1e-8}                                       , "cuda:0"),
#    ("MRK-6", "sgd",  {"lr" : 1e-8, "momentum" : 0.1}                     , "cuda:0"),
#    ("MRK-7", "sgd",  {"lr" : 1e-8, "momentum" : 0.1, "dampening" : 0.1 } , "cuda:0"),
#    ("MRK-8", "sgd",  {"lr" : 1e-8, "momentum" : 0.1, "dampening" : 0.01} , "cuda:0")
]

graphs = [
    ("GraphTruthJets"    , "BSM4Tops"),
    ("GraphTruthJetsNoNu", "BSM4Tops"),
    ("GraphJets"         , "BSM4Tops"),
    ("GraphDetector"     , "BSM4Tops")
]

kfolds = 10
sample_path = "/home/tnom6927/Downloads/test/*"

for gr in graphs:
    gr_name, ev_name = gr
    path = "gnn-results/" + gr_name
    pathlib.Path(path).mkdir(parents = True, exist_ok = True)
    for k in range(1, kfolds+1):
        f = open("./configs/mc16/template-config.yaml", "rb")
        data = yaml.load(f, Loader = yaml.CLoader)
        f.close()

        data["base"]["graph"] = gr_name
        data["base"]["event"] = ev_name
        data["base"]["project-name"] = gr_name
        data["base"]["sample-path"] = sample_path
        data["base"]["campaign"] = "mc16"
        data["base"]["kfold"] = [k]
        data["base"]["samples"] = {f : l for f, l in samples}

        dumps = []
        for pr in params:
            name, optim, para, dev = pr
            params_header = dict(data["<name>"])
            params_dump = {name : params_header}
            params_dump[name]["device"] = dev
            params_dump[name]["optimizer"] = {"Optimizer" : optim}
            params_dump[name]["optimizer"] |= para
            dumps.append(params_dump)

        del data["<name>"]
        for i in dumps: data |= i
        s = yaml.dump(data).encode("utf-8")
        f = open(path + "/config_k-" + str(k) + ".yaml", "wb")
        f.write(s)
        f.close()

