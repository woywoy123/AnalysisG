import yaml
import pathlib

params = [
    ("MRK-1", "adam", {"lr" : 1e-6}),
    ("MRK-2", "adam", {"lr" : 1e-8}),
    ("MRK-3", "adam", {"lr" : 1e-8, "amsgrad" : True}),
    ("MRK-4", "sgd", {"lr" : 1e-6}),
    ("MRK-5", "sgd", {"lr" : 1e-8}),
    ("MRK-6", "sgd", {"lr" : 1e-6, "momentum" : 0.1}),
    ("MRK-7", "sgd", {"lr" : 1e-6, "momentum" : 0.1, "dampening" : 0.01})
]

graphs = [
    ("GraphTruthJets", False),
    ("GraphTruthJetNoNu", True),
    ("GraphJets", False),
    ("GraphDetector", True)
]

kfolds = 10
src = "./"
sample_pth = ""
result_path = "./gnn-results/"
user_path = "$PWD"
scripts = []
for gr in graphs:
    gr_name, reco = gr
    for pr in params:
        name, optim, parm = pr
        pth = gr_name + "-" + name
        pathlib.Path(result_path + pth).mkdir(parents = True, exist_ok = True)
        for k in range(1, kfolds+1):

            f = open(src + "config.yaml", "rb")
            data = yaml.load(f, Loader = yaml.CLoader)
            data["training"]["io"]["project-name"] = "rnn-" + gr_name
            data["training"]["io"]["output-path"] = result_path
            data["training"]["run-name"] = name
            data["training"]["graph"] = gr_name
            data["training"]["model"]["extra-flags"]["NuR"] = reco
            data["training"]["sample-path"] = data["training"]["sample-path"].replace("<user>", sample_pth)

            data["training"]["optimizer"]["Optimizer"] = optim
            for p in parm: data["training"]["optimizer"][p] = parm[p]
            data["training"]["train"]["kfold"] = [k]
            f_ = open(result_path + pth + "/kfold-" + str(k) + ".yaml", "wb")
            f_.write(yaml.dump(data).encode("utf-8"))
            f_.close()

            fx = open(src + "runner.sh", "rb")
            scrpt = fx.read().decode("utf-8")
            fx.close()

            scrpt = scrpt.replace("<path>", user_path)
            scrpt = scrpt.replace("<res-path>", src)
            scrpt = scrpt.replace("<config-file>", result_path + pth + "/kfold-" + str(k) + ".yaml")
            scrpt = scrpt.replace("<src>", src)
            f_ = open(result_path + pth + "/kfold-" + str(k) + "-runner.sh", "wb")
            f_.write(scrpt.encode("utf-8"))
            f_.close()

            fx = open(src + "job_submit.sh", "rb")
            jb = fx.read().decode("utf-8")
            fx.close()

            jb = jb.replace("<name-this>", gr_name + "-" + name + "-kf-" + str(k))
            jb = jb.replace("<script-path>", result_path + pth + "/kfold-" + str(k) + "-runner.sh")
            f_ = open(result_path + pth + "/kfold-" + str(k) + "-job_submit.sh", "wb")
            f_.write(jb.encode("utf-8"))
            f_.close()
            scripts.append(result_path + pth + "/kfold-" + str(k) + "-job_submit.sh")

sk = open("batch.sh", "wb")
sk.write(("\n".join(["sbatch " + i for i in scripts]+[""])).encode("utf-8"))
sk.close()
