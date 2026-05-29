from training.config import MatrixCfg
from training.main import PipeLine
import argparse
import os

parse = argparse.ArgumentParser(description = "Configuration YAML file")
parse.add_argument("--config", action = "store", dest = "config")
cfg = MatrixCfg(None, None, None, None).import_yaml(parse.parse_args().config)

pl = PipeLine()

val = cfg.get_value("base", "project", "input") 
if val is not None: pl.sample.fetch_path(val)

val = cfg.get_value("base", "project", "output")
if val is not None: pl.OutputPath = val

val = cfg.get_value("base", "project", "debug")
if val is not None: pl.DebugMode = val

val = cfg.get_value("base", "project", "threads")
if val is not None: pl.Threads = val

val = cfg.get_value("base", "project", "intra")
if val is not None: pl.IntraThreads = val


val = cfg.get_value("base", "event", "name") 
if val is not None: pl.sample.event.name = val

val = cfg.get_value("base", "graph", "name")
if val is not None: pl.sample.graph.name = val

val = cfg.get_value("base", "graph", "path")
if val is not None: pl.sample.graph.GraphCache = val

val = cfg.get_value("base", "graph", "build")
if val is not None: pl.sample.graph.BuildCache = val


val = cfg.get_value("base", "training", "kfold")
if val is not None: pl.training.kFold = val

val = cfg.get_value("base", "training", "epochs")
if val is not None: pl.training.Epochs = val

val = cfg.get_value("base", "training", "batches")
if val is not None: pl.training.BatchSize = val

val = cfg.get_value("base", "training", "kfolds")
if val is not None: pl.training.kFolds = val

val = cfg.get_value("base", "training", "splits")
if val is not None: pl.training.TrainSize = val

val = cfg.get_value("base", "training", "dataset")
if val is not None: pl.training.TrainDataset = val


val = cfg.get_value("base", "modes", "training")
if val is not None: pl.training.Training = val

val = cfg.get_value("base", "modes", "validation")
if val is not None: pl.training.Validation = val

val = cfg.get_value("base", "modes", "evaluation")
if val is not None: pl.training.Evaluation = val

val = cfg.get_value("base", "modes", "continue")
if val is not None: pl.training.ContinueTraining = val


val = cfg.get_value("base", "plotting", "nbins")
if val is not None: pl.cosmetic.nBins = val

val = cfg.get_value("base", "plotting", "range")
if val is not None: pl.cosmetic.MaxRange = val

val = cfg.get_value("base", "plotting", "variables")
if val is not None: pl.cosmetic.VarPt     = val[0]
if val is not None: pl.cosmetic.VarEta    = val[1]
if val is not None: pl.cosmetic.VarPhi    = val[2]
if val is not None: pl.cosmetic.VarEnergy = val[3]

val = cfg.get_value("base", "plotting", "target")
if val is not None: pl.cosmetic.Targets = val

val = cfg.get_value("base", "plotting", "logy")
if val is not None: pl.cosmetic.SetLogY = val

val = cfg.get_value("base", "models")
for i in val:
    model = pl.ConfigModel(i, cfg.get_value(i, "base", "model"), cfg.get_value(i, "optimizer", "algorithm"))

    keys = cfg.get_value(i, "optimizer", "parameters")
    for k in keys: setattr(model.optimizer, k, keys[k]) 
    model.device = cfg.get_value(i, "base", "device")
        
    keys = cfg.get_value(i, "scheduler", "parameters")
    for k in keys: setattr(model.scheduler, k, keys[k]) 

    keys = cfg.get_value(i, "features", "edge")
    feats = [k for k in keys if not isinstance(keys[k], dict)]
    for k in feats: model.EdgeFeature(k)

    keys = cfg.get_value(i, "features", "node")
    feats = [k for k in keys if not isinstance(keys[k], dict)]
    for k in feats: model.NodeFeature(k)

    keys = cfg.get_value(i, "features", "graph")
    feats = [k for k in keys if not isinstance(keys[k], dict)]
    for k in feats: model.GraphFeature(k)

    keys = cfg.get_value(i, "features", "edge")
    feats = [k for k in keys if isinstance(keys[k], dict)]
    for k in feats: model.EdgeFeature(k, model.ConfigLoss(*keys[k]))

    keys = cfg.get_value(i, "features", "node")
    feats = [k for k in keys if isinstance(keys[k], dict)]
    for k in feats: model.NodeFeature(k, model.ConfigLoss(*keys[k]))

    keys = cfg.get_value(i, "features", "graph")
    feats = [k for k in keys if isinstance(keys[k], dict)]
    for k in feats: model.GraphFeature(k, model.ConfigLoss(*keys[k]))

pl.Build()

